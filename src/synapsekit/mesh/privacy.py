"""Privacy filters for local knowledge mesh indexing."""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MESH_IGNORE = Path.home() / ".synapsekit" / "mesh.ignore"

DEFAULT_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".tox",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)
DEFAULT_SECRET_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*credentials*",
    "*secret*",
    "*secrets*",
    "*token*",
    "*.key",
    "*.pem",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
)

_SENSITIVE_LINE_RE = re.compile(
    r"(?im)^.*\b(api[_-]?key|secret|token|password|private[_-]?key)\b\s*[:=].*$"
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class PrivacyDecision:
    """Decision returned by ``MeshPrivacyFilter.evaluate``."""

    allowed: bool
    reason: str | None = None


@dataclass
class MeshPrivacyFilter:
    """Gitignore-style filter for files that should never be indexed."""

    ignore_file: str | Path | None = DEFAULT_MESH_IGNORE
    extra_patterns: list[str] = field(default_factory=list)
    skip_dirs: frozenset[str] = DEFAULT_SKIP_DIRS
    secret_patterns: tuple[str, ...] = DEFAULT_SECRET_PATTERNS

    def __post_init__(self) -> None:
        self._patterns = [*self.extra_patterns, *self._read_ignore_file()]

    @property
    def patterns(self) -> tuple[str, ...]:
        """All configured user ignore patterns."""

        return tuple(self._patterns)

    def evaluate(self, path: str | Path, root: str | Path | None = None) -> PrivacyDecision:
        """Return whether ``path`` is allowed to be indexed."""

        candidate = Path(path)
        name = candidate.name
        if any(part in self.skip_dirs for part in candidate.parts):
            return PrivacyDecision(False, "ignored directory")
        if any(fnmatch.fnmatch(name, pattern) for pattern in self.secret_patterns):
            return PrivacyDecision(False, "secret-like filename")

        rel = self._relative_key(candidate, root)
        for pattern in self._patterns:
            if self._matches_pattern(rel, name, pattern):
                return PrivacyDecision(False, f"matched ignore pattern: {pattern}")
        return PrivacyDecision(True)

    def allows(self, path: str | Path, root: str | Path | None = None) -> bool:
        """Return ``True`` when ``path`` can be indexed."""

        return self.evaluate(path, root).allowed

    def redact_text(self, text: str) -> str:
        """Redact sensitive-looking lines before the text enters the mesh."""

        text = _PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", text)
        return _SENSITIVE_LINE_RE.sub("[REDACTED SECRET]", text)

    def _read_ignore_file(self) -> list[str]:
        if self.ignore_file is None:
            return []
        path = Path(self.ignore_file).expanduser()
        if not path.exists():
            return []
        patterns: list[str] = []
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(line)
        return patterns

    @staticmethod
    def _relative_key(path: Path, root: str | Path | None) -> str:
        if root is None:
            return path.as_posix()
        try:
            return path.resolve().relative_to(Path(root).expanduser().resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    @staticmethod
    def _matches_pattern(relative_path: str, name: str, pattern: str) -> bool:
        normalized = pattern.replace(os.sep, "/")
        if normalized.endswith("/"):
            normalized = normalized.rstrip("/")
        if normalized.startswith("/"):
            normalized = normalized.lstrip("/")
            return fnmatch.fnmatch(relative_path, normalized)
        if "/" in normalized:
            return fnmatch.fnmatch(relative_path, normalized)
        return fnmatch.fnmatch(name, normalized) or fnmatch.fnmatch(relative_path, normalized)
