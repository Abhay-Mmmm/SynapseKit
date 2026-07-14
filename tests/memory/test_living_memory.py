from __future__ import annotations

import json
from pathlib import Path

import pytest

from synapsekit.memory import (
    FileDiffEngine,
    LivingMemory,
    MemoryFileRouter,
    MemoryPatch,
    MemoryPIIFilter,
    OccurrenceTracker,
    PatchStore,
)


class MockLLM:
    """Mock LLM that returns a pre-configured response."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[str] = []

    async def agenerate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response_text

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response_text


def test_memory_patch_sign_verify() -> None:
    patch = MemoryPatch(
        file_path="CLAUDE.md",
        before_content="Hello",
        after_content="Hello World",
        unified_diff="--- \n+++ \n@@ -1 +1 @@\n-Hello\n+Hello World\n",
        rationale="Add World",
    )
    secret = "test-secret"
    sig = patch.sign(secret)
    assert sig != ""
    assert patch.signature == sig
    assert patch.verify(secret) is True
    assert patch.verify("wrong-secret") is False

    # Modify content and verify it fails
    patch.after_content = "Hello Modified"
    assert patch.verify(secret) is False


def test_diff_engine_unified_diff() -> None:
    before = "line1\nline2\n"
    after = "line1\nline2 changed\n"
    diff = FileDiffEngine.generate_unified_diff(before, after, "test.txt")
    assert "line2" in diff
    assert "line2 changed" in diff

    stats = FileDiffEngine.count_changed_lines(diff)
    assert stats["added"] == 1
    assert stats["removed"] == 1


def test_diff_engine_validation(tmp_path: Path) -> None:
    file_path = tmp_path / "test.md"
    content = "Hello World\n"
    file_path.write_text(content, encoding="utf-8")

    # Content matches exactly
    ok, _ = FileDiffEngine.validate_patch_applicable(file_path, content)
    assert ok is True

    # Content matches with trailing whitespace diff
    ok, _ = FileDiffEngine.validate_patch_applicable(file_path, "Hello World \n")
    assert ok is True

    # Content diverged
    ok, _ = FileDiffEngine.validate_patch_applicable(file_path, "Diverged content")
    assert ok is False


def test_patch_store(tmp_path: Path) -> None:
    store_file = tmp_path / "patches.jsonl"
    store = PatchStore(store_file)

    patch = MemoryPatch(
        file_path="CLAUDE.md",
        before_content="A",
        after_content="B",
        unified_diff="...",
        rationale="Change A to B",
    )
    store.save(patch)

    # Reload store
    store2 = PatchStore(store_file)
    retrieved = store2.get(patch.patch_id)
    assert retrieved is not None
    assert retrieved.rationale == "Change A to B"
    assert retrieved.status == "pending"

    # Update status
    retrieved.status = "applied"
    store2.update(retrieved)

    # Check update persisted
    store3 = PatchStore(store_file)
    retrieved2 = store3.get(patch.patch_id)
    assert retrieved2 is not None
    assert retrieved2.status == "applied"


def test_patch_store_update_preserves_secret_signature(tmp_path: Path) -> None:
    """Regression test for PatchStore.update() clobbering secret-signed signatures.

    Bug: update() unconditionally called patch.sign() with no secret, which
    overwrote any secret-signed signature set by the caller with an
    empty-secret one before persisting, breaking patch.verify(secret) for
    applied/reverted patches and letting forged signatures verify.
    Root cause: patch.sign() call inside PatchStore.update().
    Fix: removed the redundant sign() call from update() -- callers already
    sign with the correct secret before calling update().
    """
    store_file = tmp_path / "patches.jsonl"
    store = PatchStore(store_file)

    patch = MemoryPatch(
        file_path="CLAUDE.md",
        before_content="A",
        after_content="B",
        unified_diff="...",
        rationale="Change A to B",
    )
    secret = "super-secret"
    store.save(patch)

    # Simulate what LivingMemory.apply/revert do: sign with the real
    # secret immediately before calling update().
    patch.status = "applied"
    patch.sign(secret)
    store.update(patch)

    # The persisted patch must still verify against the real secret.
    reloaded = PatchStore(store_file).get(patch.patch_id)
    assert reloaded is not None
    assert reloaded.status == "applied"
    assert reloaded.verify(secret) is True

    # Negative case: verifying with the wrong (or empty) secret must fail.
    assert reloaded.verify("wrong-secret") is False
    assert reloaded.verify("") is False


def test_occurrence_tracker(tmp_path: Path) -> None:
    tracker_file = tmp_path / "occurrences.json"
    tracker = OccurrenceTracker(tracker_file)

    fact = "user_prefers_concise_answers"
    tracker.record_occurrence(fact, "session_1", "evidence 1")
    assert tracker.has_reached_threshold(fact, 3) is False

    tracker.record_occurrence(fact, "session_2", "evidence 2")
    tracker.record_occurrence(fact, "session_3", "evidence 3")
    assert tracker.has_reached_threshold(fact, 3) is True

    # Reload tracker
    tracker2 = OccurrenceTracker(tracker_file)
    assert tracker2.has_reached_threshold(fact, 3) is True
    assert tracker2.get_count(fact) == 3


def test_pii_filter() -> None:
    pii_filter = MemoryPIIFilter()

    # Clean text
    res = pii_filter.filter_content("Nothing sensitive here.")
    assert res.is_clean is True

    # Text containing email and credit card
    res2 = pii_filter.filter_content("Email me at test@example.com or use card 1234-5678-1234-5678")
    assert res2.is_clean is False
    assert "[REDACTED_EMAIL]" in res2.filtered_content
    assert "[REDACTED_CC]" in res2.filtered_content
    assert "email" in res2.redaction_types
    assert "credit_card" in res2.redaction_types


def test_pii_filter_detects_api_key_not_covered_by_pii_detector() -> None:
    """Regression test for api_key detection being silently dropped.

    Bug: MemoryPIIFilter.__init__ built self._detector = PIIDetector(detect=...)
    filtered to only types present in PIIDetector._PATTERNS (email, phone,
    ssn, credit_card, ip_address). filter_content() gated entirely on
    self._detector.check(content).passed and returned is_clean=True
    immediately when that passed -- so content containing ONLY an api_key
    (a type covered by MemoryPIIFilter._REDACTION_PATTERNS but not by
    PIIDetector) was never flagged or redacted.
    Root cause: early return based solely on the underlying PIIDetector's
    check, ignoring active_types not supported by PIIDetector.
    Fix: filter_content() now also checks _compiled patterns not covered
    by PIIDetector._PATTERNS before deciding content is clean.
    """
    pii_filter = MemoryPIIFilter(detect=["api_key"])

    # Content with ONLY an api_key-shaped string, no email/phone/ssn/etc.
    content = "Here is my token: sk-abcdefghijklmnopqrstuvwx1234"
    res = pii_filter.filter_content(content)

    assert res.is_clean is False
    assert "[REDACTED_KEY]" in res.filtered_content
    assert "api_key" in res.redaction_types


def test_pii_filter_api_key_mixed_with_supported_type_no_regression() -> None:
    """Mixing api_key with a PIIDetector-supported type must still redact both."""
    pii_filter = MemoryPIIFilter(detect=["email", "api_key"])

    content = "Contact test@example.com, token sk-abcdefghijklmnopqrstuvwx1234"
    res = pii_filter.filter_content(content)

    assert res.is_clean is False
    assert "[REDACTED_EMAIL]" in res.filtered_content
    assert "[REDACTED_KEY]" in res.filtered_content
    assert "email" in res.redaction_types
    assert "api_key" in res.redaction_types


def test_pii_filter_redact_false_reports_clean_type_name() -> None:
    """Regression test for violation-type parsing bug in filter_content(redact=False).

    Bug: redaction_types was computed with
    v.split("(")[1].rstrip(")").split(":")[0] on strings like
    "PII detected (email): 2 instance(s)". Tracing through:
    split("(")[1] -> "email): 2 instance(s)"; rstrip(")") strips the
    trailing ")" from "instance(s)" giving "email): 2 instance(s";
    split(":")[0] -> "email)" -- includes a stray ")".
    Fix: parse with re.search(r'\\(([\\w_]+)\\)', v).group(1) instead.
    """
    pii_filter = MemoryPIIFilter(redact=False)

    res = pii_filter.filter_content("Email me at test@example.com")
    assert res.is_clean is False
    assert "email" in res.redaction_types
    assert "email)" not in res.redaction_types
    # Content must be unmodified when redact=False.
    assert res.filtered_content == res.original_content


def test_file_router() -> None:
    router = MemoryFileRouter(
        path_map={"user": "user_prefs.md", "feedback": "feedback.md"},
        primary_path="CLAUDE.md",
    )

    # Categorization based on signals
    assert router.categorize("The user prefers tabs over spaces.") == "user"
    assert router.categorize("That was a mistake, let's fix it.") == "feedback"
    assert router.categorize("The database architecture is PostgreSQL.") == "project"
    assert router.categorize("Random fact about spaceships.") == "general"

    managed = ["CLAUDE.md", "user_prefs.md", "project_info.md"]
    # Path resolution
    assert router.resolve_target_path("user", managed) == "user_prefs.md"
    assert router.resolve_target_path("project", managed) == "project_info.md"
    assert router.resolve_target_path("general", managed) == "CLAUDE.md"


@pytest.mark.asyncio
async def test_living_memory_orchestration(tmp_path: Path) -> None:
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("# Core Memory\n", encoding="utf-8")

    # Setup proposer response
    proposals = [
        {
            "file_path": str(claude_md),
            "section": "Core Memory",
            "fact_key": "user_prefers_pytest",
            "proposed_addition": "- The user prefers pytest for tests.",
            "rationale": "Add pytest preference",
            "evidence": "Use pytest always",
        }
    ]
    proposer = MockLLM(json.dumps(proposals))

    # LivingMemory instance
    lm = LivingMemory(
        paths=[str(claude_md)],
        proposer=proposer,
        require_approval=True,
        store_path=str(tmp_path / "patches.jsonl"),
        occurrence_path=str(tmp_path / "occurrences.json"),
        occurrence_threshold=1,  # Lower threshold for testing
    )

    # 1. Propose patch from session
    patches = await lm.propose_from_session("session_123", transcript="Use pytest always")
    assert len(patches) == 1
    patch = patches[0]
    assert patch.status == "pending"
    assert patch.file_path == str(claude_md)
    assert "- The user prefers pytest for tests." in patch.after_content

    # 2. Check pending list
    pending = lm.pending_patches()
    assert len(pending) == 1
    assert pending[0].patch_id == patch.patch_id

    # 3. Apply patch
    applied = lm.apply(patch.patch_id)
    assert applied.status == "applied"
    assert (
        claude_md.read_text(encoding="utf-8")
        == "# Core Memory\n- The user prefers pytest for tests.\n"
    )

    # 4. Revert patch
    reverted = lm.revert(patch.patch_id)
    assert reverted.status == "reverted"
    assert claude_md.read_text(encoding="utf-8") == "# Core Memory\n"
