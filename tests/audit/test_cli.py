"""``synapsekit audit verify`` / ``synapsekit audit replay`` CLI entry points."""

from __future__ import annotations

import pytest

from synapsekit.cli.main import main


class TestAuditCLI:
    def test_verify_exits_zero_for_a_valid_bundle(self, bundle_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["audit", "verify", str(bundle_path)])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "Verdict: MATCH" in out

    def test_verify_exits_nonzero_for_a_tampered_bundle(self, tmp_path, bundle_path, capsys):
        from .conftest import (
            dump_trace_lines,
            load_trace_lines,
            read_zip_entries,
            write_zip_entries,
        )

        entries = read_zip_entries(bundle_path)
        records = load_trace_lines(entries)
        records[0]["hash"] = "0" * 64
        dump_trace_lines(entries, records)
        tampered = tmp_path / "tampered.zip"
        write_zip_entries(tampered, entries)

        with pytest.raises(SystemExit) as exc_info:
            main(["audit", "verify", str(tampered)])
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "DRIFT" in out

    def test_verify_json_output_format(self, bundle_path, capsys):
        with pytest.raises(SystemExit):
            main(["audit", "verify", str(bundle_path), "--format", "json"])
        out = capsys.readouterr().out
        assert '"verdict": "MATCH"' in out

    def test_replay_exits_zero_for_a_valid_bundle(self, bundle_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["audit", "replay", str(bundle_path)])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "REPLAY OK" in out

    def test_replay_refuses_to_run_on_a_bundle_that_fails_verification(
        self, tmp_path, bundle_path, capsys
    ):
        from .conftest import (
            dump_trace_lines,
            load_trace_lines,
            read_zip_entries,
            write_zip_entries,
        )

        entries = read_zip_entries(bundle_path)
        records = load_trace_lines(entries)
        records[0]["hash"] = "0" * 64
        dump_trace_lines(entries, records)
        tampered = tmp_path / "tampered.zip"
        write_zip_entries(tampered, entries)

        with pytest.raises(SystemExit) as exc_info:
            main(["audit", "replay", str(tampered)])
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "Cannot replay" in out

    def test_verify_without_trusted_key_reports_self_consistent_only(self, bundle_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["audit", "verify", str(bundle_path)])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "Trust:      NONE" in out

    def test_verify_with_correct_trusted_key_passes(self, tmp_path, sample_records, capsys):
        import base64

        from synapsekit.audit import SigningPolicy, export_audit_bundle

        policy = SigningPolicy.ed25519(key_id="release-key")
        path = export_audit_bundle(sample_records, policy, tmp_path / "b.zip")
        pubkey_b64 = base64.b64encode(policy.provider.public_key_bytes()).decode("ascii")

        with pytest.raises(SystemExit) as exc_info:
            main(["audit", "verify", str(path), "--trusted-key", f"release-key:{pubkey_b64}"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "Trust:      PINNED" in out

    def test_verify_with_wrong_trusted_key_fails(self, tmp_path, sample_records, capsys):
        import base64

        from synapsekit.audit import SigningPolicy, export_audit_bundle

        policy = SigningPolicy.ed25519(key_id="release-key")
        path = export_audit_bundle(sample_records, policy, tmp_path / "b.zip")
        other_pubkey_b64 = base64.b64encode(
            SigningPolicy.ed25519().provider.public_key_bytes()
        ).decode("ascii")

        with pytest.raises(SystemExit) as exc_info:
            main(["audit", "verify", str(path), "--trusted-key", f"release-key:{other_pubkey_b64}"])
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "DRIFT" in out

    def test_malformed_trusted_key_argument_raises_system_exit(self, bundle_path):
        with pytest.raises(SystemExit):
            main(["audit", "verify", str(bundle_path), "--trusted-key", "not-a-valid-entry"])
