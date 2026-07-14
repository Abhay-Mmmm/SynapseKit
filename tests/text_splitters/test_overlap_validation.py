"""Regression tests for issue 781: chunk_overlap must be smaller than chunk_size.

When chunk_overlap equals chunk_size the hard-split range step is 0 and raises
ValueError; when chunk_overlap is greater than chunk_size the step is negative and
range yields nothing, silently dropping data. The fix rejects
chunk_overlap >= chunk_size at construction time.
"""

from __future__ import annotations

import pytest

from synapsekit.text_splitters.character import CharacterTextSplitter
from synapsekit.text_splitters.recursive import RecursiveCharacterTextSplitter


@pytest.mark.parametrize("cls", [RecursiveCharacterTextSplitter, CharacterTextSplitter])
def test_overlap_equal_to_size_rejected(cls):
    with pytest.raises(ValueError, match="chunk_overlap"):
        cls(chunk_size=100, chunk_overlap=100)


@pytest.mark.parametrize("cls", [RecursiveCharacterTextSplitter, CharacterTextSplitter])
def test_overlap_greater_than_size_rejected(cls):
    with pytest.raises(ValueError, match="chunk_overlap"):
        cls(chunk_size=100, chunk_overlap=150)


@pytest.mark.parametrize("cls", [RecursiveCharacterTextSplitter, CharacterTextSplitter])
def test_valid_overlap_accepted(cls):
    splitter = cls(chunk_size=100, chunk_overlap=20)
    assert splitter.chunk_size == 100
    assert splitter.chunk_overlap == 20


def test_recursive_splitter_no_longer_loses_data_via_hard_split():
    """A no-separator string longer than chunk_size still splits without data loss."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10, chunk_overlap=3, separators=[" "]
    )
    text = "abcdefghijklmnopqrstuvwxyz"  # no separators present
    chunks = splitter.split(text)
    assert chunks  # not silently empty
    # First chunk begins at the text start; nothing is dropped from the front.
    assert chunks[0].startswith("abcde")


def test_character_splitter_no_separator_hard_split_ok():
    splitter = CharacterTextSplitter(separator="||", chunk_size=8, chunk_overlap=2)
    chunks = splitter.split("abcdefghijklmnop")
    assert len(chunks) > 1
