from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from synapsekit.mesh import GitRepoLoader, LocalMdLoader, MeshPrivacyFilter


def test_local_md_loader_preserves_headings_lines_and_privacy(tmp_path: Path) -> None:
    ignore = tmp_path / ".mesh.ignore"
    ignore.write_text("ignored.md\n", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("# Ignored\n\nDo not index.", encoding="utf-8")
    (tmp_path / "secret-notes.md").write_text("# Secret\n\npassword = hunter2", encoding="utf-8")
    doc = tmp_path / "README.md"
    doc.write_text(
        "---\ntitle: Local\n---\n# Auth\n\nAuthService uses FastAPI middleware.\n\n## Retry\n\nRetry decorator uses backoff.",
        encoding="utf-8",
    )

    loader = LocalMdLoader(tmp_path, privacy_filter=MeshPrivacyFilter(ignore_file=ignore))
    docs = loader.load()

    assert {Path(item.metadata["path"]).name for item in docs} == {"README.md"}
    assert docs[0].metadata["line_start"] == 1
    assert docs[0].metadata["headings"] == ("Auth",)
    assert any(item.metadata["headings"] == ("Auth", "Retry") for item in docs)
    assert all("password" not in item.text for item in docs)


def test_git_repo_loader_discovers_repo_docs_and_commit_subjects(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git CLI not available")

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "mesh@example.com")
    _git(repo, "config", "user.name", "Mesh Test")
    (repo / "README.md").write_text("# Mesh\n\nAuthService pattern.", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "add mesh docs")

    docs = GitRepoLoader(tmp_path, include_history=True).load()

    assert any(doc.metadata["source_type"] == "git_repo" for doc in docs)
    assert any(doc.metadata["source_type"] == "git_history" for doc in docs)
    assert any("add mesh docs" in doc.text for doc in docs)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_default_include_matches_top_level_docs_file_without_bare_md_fallback(
    tmp_path: Path,
) -> None:
    # Regression test: "docs/**/*.md" must match files directly under docs/
    # (e.g. docs/architecture.md), not just files nested in a subdirectory.
    # Using only this single pattern (no co-listed bare "*.md" fallback)
    # exercises the exact bug: globstar "**" previously required at least
    # one subdirectory segment between "docs/" and the filename.
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "architecture.md").write_text("# Architecture\n", encoding="utf-8")
    (tmp_path / "docs" / "guides").mkdir()
    (tmp_path / "docs" / "guides" / "deep-dive.md").write_text("# Deep dive\n", encoding="utf-8")
    (tmp_path / "other.md").write_text("# Not included\n", encoding="utf-8")

    loader = LocalMdLoader(tmp_path, include=["docs/**/*.md"])
    docs = loader.load()

    relative_paths = {item.metadata["relative_path"] for item in docs}
    assert relative_paths == {"docs/architecture.md", "docs/guides/deep-dive.md"}


def test_matches_include_globstar_semantics_directly() -> None:
    matches = LocalMdLoader._matches_include

    # Zero directory segments between docs/ and the filename.
    assert matches("architecture.md", "docs/architecture.md", "docs/**/*.md") is True
    # One or more nested directory segments.
    assert matches("deep-dive.md", "docs/guides/deep-dive.md", "docs/**/*.md") is True
    assert matches("deep-dive.md", "docs/a/b/deep-dive.md", "docs/**/*.md") is True
    # Different top-level directory must not match.
    assert matches("x.md", "notdocs/x.md", "docs/**/*.md") is False
    # Different extension must not match.
    assert matches("architecture.txt", "docs/architecture.txt", "docs/**/*.md") is False
