"""``synapsekit audit`` commands: verify and replay signed audit bundles."""

from __future__ import annotations

import base64
import sys
from typing import Any


def _add_trusted_key_arg(parser: Any) -> None:
    parser.add_argument(
        "--trusted-key",
        dest="trusted_keys",
        action="append",
        metavar="KEY_ID:BASE64_PUBLIC_KEY",
        default=None,
        help=(
            "Pin a public key obtained independently of the bundle (repeatable). "
            "Without this, verification only proves the bundle wasn't edited after "
            "export — NOT who produced it, since public keys are otherwise read "
            "from the bundle's own manifest."
        ),
    )


def _parse_trusted_keys(raw: list[str] | None) -> dict[str, bytes] | None:
    if not raw:
        return None
    trusted: dict[str, bytes] = {}
    for entry in raw:
        key_id, sep, b64 = entry.partition(":")
        if not sep:
            raise SystemExit(f"invalid --trusted-key {entry!r}; expected KEY_ID:BASE64_PUBLIC_KEY")
        trusted[key_id] = base64.b64decode(b64)
    return trusted


def build_audit_parser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "audit", help="Verify and replay cryptographically signed audit bundles"
    )
    audit_sub = p.add_subparsers(dest="audit_command")

    verify_cmd = audit_sub.add_parser(
        "verify", help="Verify a bundle's chain, Merkle tree, signatures, and manifest"
    )
    verify_cmd.add_argument("bundle", help="Path to the .zip audit bundle")
    verify_cmd.add_argument(
        "--format",
        dest="output_format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    _add_trusted_key_arg(verify_cmd)

    replay_cmd = audit_sub.add_parser(
        "replay", help="Reconstruct and verify provenance from a bundle"
    )
    replay_cmd.add_argument("bundle", help="Path to the .zip audit bundle")
    replay_cmd.add_argument(
        "--format",
        dest="output_format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    _add_trusted_key_arg(replay_cmd)


def run_audit(args: Any) -> None:
    if args.audit_command == "verify":
        _run_verify(args)
        return
    if args.audit_command == "replay":
        _run_replay(args)
        return
    raise SystemExit("Missing audit subcommand. Use: verify or replay")


def _run_verify(args: Any) -> None:
    from ..audit.verifier import verify

    trusted_keys = _parse_trusted_keys(getattr(args, "trusted_keys", None))
    result = verify(args.bundle, trusted_keys=trusted_keys)

    if args.output_format == "json":
        import json

        print(
            json.dumps(
                {
                    "verdict": result.verdict.value,
                    "record_count": result.record_count,
                    "batch_count": result.batch_count,
                    "schema_version": result.schema_version,
                    "trust_anchor": result.trust_anchor,
                    "errors": result.errors,
                },
                indent=2,
            )
        )
    else:
        print(f"Bundle: {args.bundle}")
        print(f"Schema version: {result.schema_version}")
        print(f"Records: {result.record_count}  Batches: {result.batch_count}")
        print(f"Verdict: {result.verdict.value}")
        if result.ok:
            # A MATCH is only reachable with a pinned trust anchor now —
            # an unpinned but self-consistent bundle is capped at
            # UNVERIFIABLE (handled below), never reported as MATCH.
            print("Chain:      OK")
            print("Merkle:     OK")
            print("Signatures: OK")
            print("Manifest:   OK")
            print("Trust:      PINNED — signatures verified against caller-supplied keys")
            print("\nMATCH — bundle is intact and was signed by a key you independently trust.")
        elif result.verdict.value == "DRIFT":
            print("\nDRIFT — evidence contradicts recorded claims (tampering detected):")
            for err in result.errors:
                print(f"  - {err}")
        elif result.trust_anchor == "none":
            # Self-consistent, but no key was pinned: the verifier caps
            # this at UNVERIFIABLE rather than MATCH, because signatures
            # were only checked against keys embedded in the bundle itself.
            print(
                "Trust:      NONE — signatures verified only against keys embedded in the bundle"
            )
            print(
                "\nUNVERIFIABLE — the bundle is internally self-consistent (it wasn't edited "
                "after export), but its signer's identity is UNAUTHENTICATED. Pass --trusted-key "
                "with an independently-obtained public key to establish authenticity and get a MATCH:"
            )
            for err in result.errors:
                print(f"  - {err}")
        else:
            print("\nUNVERIFIABLE — not enough evidence to reach a conclusion:")
            for err in result.errors:
                print(f"  - {err}")

    sys.exit(0 if result.ok else 1)


def _run_replay(args: Any) -> None:
    from ..audit.replay import ReplayEngine
    from ..audit.verifier import verify

    trusted_keys = _parse_trusted_keys(getattr(args, "trusted_keys", None))
    verification = verify(args.bundle, trusted_keys=trusted_keys)
    if not verification.ok:
        print("Cannot replay: bundle failed verification.")
        for err in verification.errors:
            print(f"  - {err}")
        sys.exit(1)

    report = ReplayEngine().replay(args.bundle)

    if args.output_format == "json":
        import json

        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "record_count": report.record_count,
                    "checked_tool_calls": report.checked_tool_calls,
                    "checked_retrievals": report.checked_retrievals,
                    "checked_decisions": report.checked_decisions,
                    "skipped_llm_calls": report.skipped_llm_calls,
                    "mismatches": [m.__dict__ for m in report.mismatches],
                },
                indent=2,
                default=str,
            )
        )
    else:
        print(f"Bundle: {args.bundle}")
        print(f"Records: {report.record_count}")
        print(f"Tool calls checked: {report.checked_tool_calls}")
        print(f"Retrievals checked: {report.checked_retrievals}")
        print(f"Decisions checked: {report.checked_decisions}")
        print(
            f"LLM calls skipped (non-deterministic, not required to match): {report.skipped_llm_calls}"
        )
        if report.ok:
            print("\nREPLAY OK — provenance reconstructed with no mismatches.")
        else:
            print("\nREPLAY MISMATCHES:")
            for m in report.mismatches:
                print(f"  - [{m.kind}] {m.event_id}: {m.reason}")

    sys.exit(0 if report.ok else 1)
