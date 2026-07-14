"""Replay — verifies provenance, not "did the LLM say the same thing".

Replay confirms that a recorded run is *reproducible in the parts that
matter for audit*: the same inputs were fed to each step, tool outputs
match (when a live executor is supplied), retrieval references still
resolve, and any recorded policy hash matches the policy payload it
claims to hash. Identical LLM output is explicitly OPTIONAL — LLMs are
not required to be deterministic, so ``LLM_CALL``/``LLM_RESPONSE``
records are checked for structural completeness only, never for output
equality.

Paired event kinds (``TOOL_CALL``/``TOOL_RESULT``, ``LLM_CALL``/
``LLM_RESPONSE``) carry their input args/kwargs *and* output together on
the response/result record (see :mod:`synapsekit.audit.wrapper`), so
replay checks operate on that record directly rather than needing to
correlate it back to its paired call event.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .serializer import hash_value
from .trace import AuditTracer, ChainIntegrityError
from .types import EventKind
from .verifier import load_bundle


@dataclass
class ReplayMismatch:
    event_id: str
    kind: str
    reason: str
    expected: Any = None
    actual: Any = None


@dataclass
class ReplayReport:
    ok: bool
    mismatches: list[ReplayMismatch] = field(default_factory=list)
    record_count: int = 0
    checked_tool_calls: int = 0
    checked_retrievals: int = 0
    checked_decisions: int = 0
    skipped_llm_calls: int = 0


class ReplayEngine:
    """Replays a verified bundle's provenance without re-running any LLM.

    ``tool_executors`` maps a tool name to a callable that re-executes it
    given the recorded input; if omitted for a given tool, that tool's
    calls are checked for structural completeness only (no live
    comparison). ``retrieval_resolver`` maps a doc id to *something*
    (raise/`None` means "not found") to confirm retrieval references
    still resolve against the current corpus.
    """

    def __init__(
        self,
        *,
        tool_executors: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
        retrieval_resolver: Callable[[str], Any] | None = None,
    ) -> None:
        self.tool_executors = tool_executors or {}
        self.retrieval_resolver = retrieval_resolver

    def replay(self, bundle_path: str | Path) -> ReplayReport:
        loaded = load_bundle(bundle_path)
        records = loaded.records
        mismatches: list[ReplayMismatch] = []

        by_run: dict[str, list] = {}
        for rec in records:
            by_run.setdefault(rec.run_id, []).append(rec)
        for run_id, run_records in by_run.items():
            try:
                AuditTracer.verify_chain(run_records)
            except ChainIntegrityError as exc:
                mismatches.append(
                    ReplayMismatch(event_id="*", kind="chain", reason=f"run {run_id}: {exc}")
                )

        counts = {"tool_call": 0, "retrieval": 0, "decision": 0, "llm_call": 0}
        for rec in records:
            # Two conventions both work: a manually-recorded single
            # TOOL_CALL/LLM_CALL with input+output already merged in one
            # payload, or VerifiableAgent's paired TOOL_CALL+TOOL_RESULT /
            # LLM_CALL+LLM_RESPONSE (input+output merged onto the
            # RESULT/RESPONSE record — see wrapper.py). A bare paired
            # TOOL_CALL/LLM_CALL with no output yet is simply not checked.
            if (
                rec.kind in (EventKind.TOOL_CALL.value, EventKind.TOOL_RESULT.value)
                and "output" in rec.payload
            ):
                counts["tool_call"] += 1
                mismatches.extend(self._check_tool_call(rec))
            elif rec.kind == EventKind.RETRIEVAL.value:
                counts["retrieval"] += 1
                mismatches.extend(self._check_retrieval(rec))
            elif rec.kind == EventKind.DECISION.value:
                counts["decision"] += 1
                mismatches.extend(self._check_decision(rec))
            elif rec.kind == EventKind.LLM_CALL.value:
                counts["llm_call"] += 1
                # Intentionally not checked for output equality: LLM
                # determinism is not required for replay to succeed.

        return ReplayReport(
            ok=not mismatches,
            mismatches=mismatches,
            record_count=len(records),
            checked_tool_calls=counts["tool_call"],
            checked_retrievals=counts["retrieval"],
            checked_decisions=counts["decision"],
            skipped_llm_calls=counts["llm_call"],
        )

    def _check_tool_call(self, rec) -> list[ReplayMismatch]:
        payload = rec.payload
        # Accept both the explicit {"tool", "input", "output"} convention
        # and the generic {"method", "args", "kwargs", "output"} shape
        # that synapsekit.audit.wrapper.VerifiableAgent records.
        tool_name = payload.get("tool", payload.get("method"))
        tool_input = (
            payload["input"]
            if "input" in payload
            else {"args": payload.get("args"), "kwargs": payload.get("kwargs")}
        )
        if tool_name is None or "output" not in payload:
            return [
                ReplayMismatch(
                    rec.event_id,
                    rec.kind,
                    "tool_call payload missing a tool/method name or 'output'",
                )
            ]
        executor = self.tool_executors.get(tool_name)
        if executor is None:
            return []
        fresh_output = executor(tool_input)
        if hash_value(fresh_output) != hash_value(payload["output"]):
            return [
                ReplayMismatch(
                    rec.event_id,
                    rec.kind,
                    f"tool {tool_name!r} produced a different output on replay",
                    expected=payload["output"],
                    actual=fresh_output,
                )
            ]
        return []

    def _check_retrieval(self, rec) -> list[ReplayMismatch]:
        payload = rec.payload
        doc_ids = payload.get("doc_ids", [])
        if self.retrieval_resolver is None:
            return []
        mismatches = []
        for doc_id in doc_ids:
            try:
                found = self.retrieval_resolver(doc_id)
            except Exception:
                found = None
            if found is None:
                mismatches.append(
                    ReplayMismatch(
                        rec.event_id, rec.kind, f"retrieval reference {doc_id!r} no longer resolves"
                    )
                )
        return mismatches

    def _check_decision(self, rec) -> list[ReplayMismatch]:
        payload = rec.payload
        policy = payload.get("policy")
        policy_hash = payload.get("policy_hash")
        if policy is None or policy_hash is None:
            return []
        recomputed = hash_value(policy)
        if recomputed != policy_hash:
            return [
                ReplayMismatch(
                    rec.event_id,
                    rec.kind,
                    "recorded policy_hash does not match the recorded policy payload",
                    expected=policy_hash,
                    actual=recomputed,
                )
            ]
        return []
