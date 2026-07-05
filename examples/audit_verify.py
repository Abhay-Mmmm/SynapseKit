"""Audit bundle verification example — run after ``examples/verifiable_agent.py``.

Demonstrates the independent-verification story: :func:`verify` only
depends on the stdlib plus ``cryptography`` (see the module docstring in
``synapsekit/audit/verifier.py``), so a compliance auditor could run
this same check on a machine with no SynapseKit installed. Also shows
replay (confirming tool outputs/retrieval refs/policy hashes) and the
equivalent CLI commands.
"""

from __future__ import annotations

import sys

from synapsekit.audit import ReplayEngine
from synapsekit.audit.verifier import verify


def main(bundle_path: str = "support_run.audit.zip") -> int:
    result = verify(bundle_path)

    print(f"Bundle:          {bundle_path}")
    print(f"Schema version:  {result.schema_version}")
    print(f"Records:         {result.record_count}")
    print(f"Signed batches:  {result.batch_count}")
    print(f"Verdict:         {result.verdict.value}")

    if not result.ok:
        print(f"\n{result.verdict.value}:")
        for error in result.errors:
            print(f"  - {error}")
        return 1

    print("\nMATCH — hash chain, Merkle roots, and signatures all check out.")

    # Replay checks provenance (inputs, tool outputs, retrieval refs,
    # policy hashes) without requiring the LLM to have said the same
    # thing twice — LLM determinism is explicitly optional.
    report = ReplayEngine().replay(bundle_path)
    print(f"\nReplay: {'OK' if report.ok else 'MISMATCHES FOUND'}")
    print(f"  tool calls checked:  {report.checked_tool_calls}")
    print(f"  retrievals checked:  {report.checked_retrievals}")
    print(f"  decisions checked:   {report.checked_decisions}")
    print(
        f"  LLM calls skipped:   {report.skipped_llm_calls} (non-deterministic, not required to match)"
    )
    for mismatch in report.mismatches:
        print(f"  - [{mismatch.kind}] {mismatch.event_id}: {mismatch.reason}")

    return 0


# Equivalent CLI:
#   synapsekit audit verify support_run.audit.zip
#   synapsekit audit replay support_run.audit.zip

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "support_run.audit.zip"
    sys.exit(main(path))
