"""Shared fixtures for the audit test suite."""

from __future__ import annotations

import base64
import json
import zipfile
from pathlib import Path

import pytest

from synapsekit.audit import AuditTracer, EventKind, SigningPolicy, export_audit_bundle
from synapsekit.audit.serializer import hash_value


def build_sample_records():
    tracer = AuditTracer(run_id="run-1")
    tracer.record(
        EventKind.LLM_CALL, {"model": "gpt-4o-mini", "prompt": "What is 2+2?", "response": "4"}
    )
    tracer.record(
        EventKind.TOOL_CALL,
        {"tool": "calculator", "input": {"expr": "2+2"}, "output": 4},
    )
    tracer.record(
        EventKind.RETRIEVAL,
        {"query": "capital of france", "doc_ids": ["doc-1", "doc-2"]},
    )
    policy = {"rule": "always_confirm"}
    tracer.record(
        EventKind.DECISION,
        {"policy": policy, "policy_hash": hash_value(policy), "decision": "proceed"},
    )
    return tracer.drain()


@pytest.fixture
def sample_records():
    return build_sample_records()


@pytest.fixture
def bundle_path(tmp_path: Path, sample_records) -> Path:
    policy = SigningPolicy.ed25519()
    return export_audit_bundle(sample_records, policy, tmp_path / "sample.audit.zip")


def read_zip_entries(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def manifest_keys_as_trusted(path: Path) -> dict[str, bytes]:
    """Pin every key the bundle's own manifest advertises.

    Verification is now capped at UNVERIFIABLE when no key is pinned (the
    security fix in verifier._apply_trust_anchor_cap), so structural /
    tamper tests that only care about hash-chain and Merkle integrity —
    not signer authenticity — pin the bundle's embedded keys to reach a
    genuine MATCH. Tamper tests still fail (DRIFT) regardless of pinning.
    """
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    # Selective-disclosure bundles carry their keys under the embedded
    # original_manifest rather than at the top level.
    keys = manifest.get("keys") or manifest.get("original_manifest", {}).get("keys", {})
    return {
        key_id: base64.b64decode(info["public_key_b64"]) for key_id, info in keys.items()
    }


def write_zip_entries(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)


def load_trace_lines(entries: dict[str, bytes]) -> list[dict]:
    return [
        json.loads(line)
        for line in entries["trace.jsonl"].decode("utf-8").splitlines()
        if line.strip()
    ]


def dump_trace_lines(entries: dict[str, bytes], records: list[dict]) -> None:
    entries["trace.jsonl"] = ("\n".join(json.dumps(r) for r in records) + "\n").encode("utf-8")
