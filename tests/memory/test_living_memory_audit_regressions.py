"""Regression test for LivingMemory auto-apply patch loss (issue #792).

When ``require_approval=False`` and the LLM proposes two additions to the same
file in one session, ``propose_from_session`` read the file once and every
proposal computed its ``after`` from that stale snapshot — so applying the
second patch overwrote the first on disk.

Hand-written proposer LLM only — no MagicMock.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synapsekit.memory import LivingMemory


class ProposerLLM:
    """Returns a fixed JSON list of proposals via async ``agenerate``."""

    def __init__(self, proposals: list[dict]) -> None:
        self._payload = json.dumps(proposals)

    async def agenerate(self, prompt: str) -> str:
        return self._payload


def _living_memory(managed_file: Path, proposals: list[dict], tmp_path: Path):
    return LivingMemory(
        paths=[str(managed_file)],
        proposer=ProposerLLM(proposals),
        require_approval=False,
        sign=False,
        store_path=str(tmp_path / "patches.jsonl"),
        occurrence_path=str(tmp_path / "occurrences.json"),
        occurrence_threshold=1,
    )


@pytest.mark.asyncio
async def test_auto_apply_preserves_earlier_patch_to_same_file(tmp_path: Path):
    """Both auto-applied additions must survive on disk.

    Fails on old code: the second patch's ``after_content`` is derived from the
    original empty file, so applying it erases the first addition.
    """
    memory_file = tmp_path / "CLAUDE.md"
    memory_file.write_text("# Memory\n", encoding="utf-8")

    proposals = [
        {
            "fact_key": "fact_one",
            "evidence": "user said one",
            "proposed_addition": "First durable fact about the user.",
            "section": "new",
            "rationale": "capture fact one",
        },
        {
            "fact_key": "fact_two",
            "evidence": "user said two",
            "proposed_addition": "Second durable fact about the user.",
            "section": "new",
            "rationale": "capture fact two",
        },
    ]

    lm = _living_memory(memory_file, proposals, tmp_path)
    patches = await lm.propose_from_session("sess-1", transcript="user chatted about two facts")

    assert len(patches) == 2
    assert all(p.status == "applied" for p in patches)

    final = memory_file.read_text(encoding="utf-8")
    assert "First durable fact about the user." in final
    assert "Second durable fact about the user." in final


@pytest.mark.asyncio
async def test_second_patch_before_content_reflects_first(tmp_path: Path):
    """The 2nd proposal's before_content must include the 1st applied change."""
    memory_file = tmp_path / "CLAUDE.md"
    memory_file.write_text("# Memory\n", encoding="utf-8")

    proposals = [
        {
            "fact_key": "a",
            "evidence": "e1",
            "proposed_addition": "AAA fact.",
            "section": "new",
            "rationale": "r",
        },
        {
            "fact_key": "b",
            "evidence": "e2",
            "proposed_addition": "BBB fact.",
            "section": "new",
            "rationale": "r",
        },
    ]

    lm = _living_memory(memory_file, proposals, tmp_path)
    patches = await lm.propose_from_session("sess-2", transcript="two facts")

    assert len(patches) == 2
    # The second patch was built on top of the first patch's applied content.
    assert "AAA fact." in patches[1].before_content
    assert "BBB fact." in patches[1].after_content
