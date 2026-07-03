"""Local markdown and git-repository loaders for the knowledge mesh."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..loaders.base import Document
from .privacy import MeshPrivacyFilter

DEFAULT_MESH_INCLUDES: tuple[str, ...] = (
    "*.md",
    "*.markdown",
    "README",
    "README.*",
    "CLAUDE.md",
    "AGENTS.md",
    "ADR*.md",
    "docs/**/*.md",
    "design/**/*.md",
    ".cursor/rules/*",
)
DEFAULT_MAX_FILE_BYTES = 2_000_000
_FRONTMATTER_RE = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n?", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class MarkdownChunk:
    """Markdown chunk with source line numbers."""

    text: str
    line_start: int
    line_end: int
    headings: tuple[str, ...]


class LocalMdLoader:
    """Load local markdown/design-doc files with mesh metadata.

    The loader preserves path, heading hierarchy, content hash, file mtime, and
    line ranges so retrieval results can cite exact local files.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        include: list[str] | tuple[str, ...] | None = None,
        privacy_filter: MeshPrivacyFilter | None = None,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        chunk_chars: int = 4_000,
        strip_frontmatter: bool = True,
    ) -> None:
        self.root = Path(root).expanduser()
        self.include = tuple(include or DEFAULT_MESH_INCLUDES)
        self.privacy_filter = privacy_filter or MeshPrivacyFilter()
        self.max_file_bytes = max_file_bytes
        self.chunk_chars = chunk_chars
        self.strip_frontmatter = strip_frontmatter

    def iter_paths(self) -> list[Path]:
        """Return indexable candidate paths below ``root``."""

        if not self.root.exists():
            return []
        if self.root.is_file():
            return [self.root] if self._include_file(self.root) else []

        paths: list[Path] = []
        for current_root, dirs, files in os.walk(self.root):
            current = Path(current_root)
            dirs[:] = [
                dirname
                for dirname in dirs
                if self.privacy_filter.allows(current / dirname, self.root)
            ]
            for filename in files:
                path = current / filename
                if self._include_file(path):
                    paths.append(path)
        return sorted(paths)

    def load(self) -> list[Document]:
        """Load matching files as chunked ``Document`` objects."""

        docs: list[Document] = []
        for path in self.iter_paths():
            docs.extend(self._load_file(path))
        return docs

    def _include_file(self, path: Path) -> bool:
        if not self.privacy_filter.allows(path, self.root):
            return False
        if not path.is_file():
            return False
        try:
            stat = path.stat()
        except OSError:
            return False
        if stat.st_size > self.max_file_bytes:
            return False
        relative = self._relative(path)
        return any(self._matches_include(path.name, relative, pattern) for pattern in self.include)

    def _load_file(self, path: Path) -> list[Document]:
        try:
            raw = path.read_bytes()
        except OSError:
            return []
        if b"\x00" in raw[:4096]:
            return []

        text = raw.decode("utf-8", errors="replace")
        content_hash = hashlib.sha256(raw).hexdigest()
        if self.strip_frontmatter:
            text = _FRONTMATTER_RE.sub("", text, count=1)
        text = self.privacy_filter.redact_text(text)

        stat = path.stat()
        relative = self._relative(path)
        chunks = split_markdown(text, chunk_chars=self.chunk_chars)
        documents: list[Document] = []
        for index, chunk in enumerate(chunks):
            chunk_id = stable_chunk_id(str(path.resolve()), content_hash, index)
            metadata = {
                "source": str(path),
                "path": str(path),
                "relative_path": relative,
                "repo_root": str(self.root),
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "headings": tuple(chunk.headings),
                "content_hash": content_hash,
                "chunk_id": chunk_id,
                "chunk_index": index,
                "mtime_ns": stat.st_mtime_ns,
                "size_bytes": stat.st_size,
                "source_type": "local_markdown",
            }
            documents.append(Document(text=chunk.text, metadata=metadata))
        return documents

    def _relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    @staticmethod
    def _matches_include(name: str, relative_path: str, pattern: str) -> bool:
        normalized = pattern.replace(os.sep, "/")
        return (
            Path(relative_path).match(normalized)
            or Path(name).match(normalized)
            or re.fullmatch(normalized.replace("*", ".*"), relative_path) is not None
        )


class GitRepoLoader:
    """Discover local git repos and load their mesh-relevant docs.

    This loader uses the git CLI only for metadata/history when available. File
    discovery and reading remain dependency-free.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        include: list[str] | tuple[str, ...] | None = None,
        privacy_filter: MeshPrivacyFilter | None = None,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        include_history: bool = True,
        history_limit: int = 50,
    ) -> None:
        self.root = Path(root).expanduser()
        self.include = tuple(include or DEFAULT_MESH_INCLUDES)
        self.privacy_filter = privacy_filter or MeshPrivacyFilter()
        self.max_file_bytes = max_file_bytes
        self.include_history = include_history
        self.history_limit = history_limit

    def discover_repos(self) -> list[Path]:
        """Return git repository roots under ``root``."""

        if not self.root.exists():
            return []
        if (self.root / ".git").exists():
            return [self.root]

        repos: list[Path] = []
        for current_root, dirs, _files in os.walk(self.root):
            current = Path(current_root)
            dirs[:] = [
                dirname
                for dirname in dirs
                if self.privacy_filter.allows(current / dirname, self.root)
            ]
            if ".git" in dirs or (current / ".git").exists():
                repos.append(current)
                dirs[:] = []
        return sorted(dict.fromkeys(repos))

    def load(self) -> list[Document]:
        """Load documentation and recent commit subjects from discovered repos."""

        docs: list[Document] = []
        for repo_root in self.discover_repos():
            docs.extend(self._load_repo(repo_root))
        return docs

    def _load_repo(self, repo_root: Path) -> list[Document]:
        loader = LocalMdLoader(
            repo_root,
            include=self.include,
            privacy_filter=self.privacy_filter,
            max_file_bytes=self.max_file_bytes,
        )
        docs = loader.load()
        commit = _git_output(repo_root, ["rev-parse", "HEAD"])
        for doc in docs:
            doc.metadata["source_type"] = "git_repo"
            doc.metadata["repo_root"] = str(repo_root)
            if commit:
                doc.metadata["git_commit"] = commit

        if self.include_history:
            history = self._history_document(repo_root, commit)
            if history is not None:
                docs.append(history)
        return docs

    def _history_document(self, repo_root: Path, commit: str | None) -> Document | None:
        subjects = _git_output(
            repo_root,
            ["log", f"--max-count={self.history_limit}", "--pretty=format:%h %s"],
        )
        if not subjects:
            return None
        digest = hashlib.sha256(subjects.encode("utf-8")).hexdigest()
        metadata = {
            "source": str(repo_root / ".git" / "commit-subjects"),
            "path": str(repo_root / ".git" / "commit-subjects"),
            "relative_path": ".git/commit-subjects",
            "repo_root": str(repo_root),
            "line_start": 1,
            "line_end": len(subjects.splitlines()),
            "headings": ("Git history",),
            "content_hash": digest,
            "chunk_id": stable_chunk_id(str(repo_root), digest, 0),
            "chunk_index": 0,
            "source_type": "git_history",
        }
        if commit:
            metadata["git_commit"] = commit
        return Document(text=subjects, metadata=metadata)


def split_markdown(text: str, *, chunk_chars: int = 4_000) -> list[MarkdownChunk]:
    """Split markdown by heading sections and stable line ranges."""

    lines = text.splitlines()
    if not lines:
        return []

    sections: list[tuple[int, int, tuple[str, ...]]] = []
    heading_stack: list[str] = []
    section_start = 1
    section_headings: tuple[str, ...] = ()

    for line_no, line in enumerate(lines, start=1):
        match = _HEADING_RE.match(line)
        if not match:
            continue
        if line_no > section_start:
            sections.append((section_start, line_no - 1, section_headings))
        level = len(match.group(1))
        title = match.group(2).strip()
        heading_stack = heading_stack[: level - 1]
        heading_stack.append(title)
        section_start = line_no
        section_headings = tuple(heading_stack)

    sections.append((section_start, len(lines), section_headings))
    chunks: list[MarkdownChunk] = []
    for start, end, headings in sections:
        section_lines = lines[start - 1 : end]
        if not "\n".join(section_lines).strip():
            continue
        chunks.extend(_split_section(section_lines, start, headings, chunk_chars))
    return chunks


def stable_chunk_id(path: str, content_hash: str, chunk_index: int) -> str:
    """Return a stable ID for a file chunk."""

    digest = hashlib.sha256(f"{path}:{content_hash}:{chunk_index}".encode()).hexdigest()
    return f"mesh_{digest[:24]}"


def _split_section(
    lines: list[str],
    line_start: int,
    headings: tuple[str, ...],
    chunk_chars: int,
) -> list[MarkdownChunk]:
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    chunks: list[MarkdownChunk] = []
    start_offset = 0
    current: list[str] = []
    current_chars = 0

    for offset, line in enumerate(lines):
        pending_chars = len(line) + 1
        if current and current_chars + pending_chars > chunk_chars:
            chunks.append(
                MarkdownChunk(
                    text="\n".join(current).strip(),
                    line_start=line_start + start_offset,
                    line_end=line_start + offset - 1,
                    headings=headings,
                )
            )
            current = []
            current_chars = 0
            start_offset = offset
        current.append(line)
        current_chars += pending_chars

    if current:
        chunks.append(
            MarkdownChunk(
                text="\n".join(current).strip(),
                line_start=line_start + start_offset,
                line_end=line_start + len(lines) - 1,
                headings=headings,
            )
        )
    return [chunk for chunk in chunks if chunk.text]


def _git_output(repo_root: Path, args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None
