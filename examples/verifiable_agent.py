"""Verifiable audit trail example — wrap an agent, sign, export, verify.

Wraps a plain agent object with :class:`~synapsekit.audit.VerifiableAgent`
so every call is captured into a hash-chained trace, batch-signs the
trace with Ed25519, redacts PII before it's ever hashed, and exports a
portable ``.zip`` bundle that can be verified independently — see
``examples/audit_verify.py`` for the verification side.
"""

from __future__ import annotations

import asyncio

from synapsekit.audit import (
    AuditTracer,
    EventKind,
    PIIRedactor,
    SigningPolicy,
    VerifiableAgent,
    export_audit_bundle,
)


class SupportAgent:
    """A plain agent — no SynapseKit imports, no audit-specific code at all."""

    async def retrieve(self, query: str) -> list[str]:
        return [f"kb-article-about-{query.replace(' ', '-')}"]

    async def generate(self, prompt: str) -> str:
        return f"Here's what I found for: {prompt}"

    async def call_tool(self, name: str, **kwargs: object) -> str:
        if name == "send_email":
            return f"sent to {kwargs.get('to')}"
        raise ValueError(f"unknown tool: {name}")


async def main() -> None:
    tracer = AuditTracer()
    # PII is redacted BEFORE anything is hashed — see redact.py's
    # docstring for why the ordering matters and can't be reversed later.
    redactor = PIIRedactor()

    agent = VerifiableAgent(SupportAgent(), tracer=tracer, redactor=redactor)

    docs = await agent.retrieve("refund policy")
    answer = await agent.generate(f"Summarize: {docs}")
    await agent.call_tool("send_email", to="alice@example.com", body=answer)

    print(f"Captured {len(tracer)} audit records for this run.")

    records = tracer.drain()
    policy = SigningPolicy.ed25519()  # generates a fresh Ed25519 keypair
    bundle_path = export_audit_bundle(records, policy, "support_run.audit.zip")
    print(f"Exported signed bundle: {bundle_path}")

    for record in records:
        print(f"  [{record.kind}] {record.event_id[:8]} parent={str(record.parent_id or '-')[:8]}")

    tool_calls = [r for r in records if r.kind == EventKind.TOOL_CALL.value]
    print(
        f"\n{len(tool_calls)} tool call(s) recorded; PII in their payloads has already been redacted."
    )


if __name__ == "__main__":
    asyncio.run(main())
