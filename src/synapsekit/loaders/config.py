from __future__ import annotations

import asyncio
import configparser
import math
import os
import re

from .base import Document

_SENSITIVE_KEYWORDS = {"password", "secret", "token", "api_key", "key", "auth"}
_SUPPORTED_EXTENSIONS = {".env", ".ini", ".cfg", ".toml"}

# A URL/DSN carrying inline credentials, e.g. postgres://user:pass@host:5432/db
# or redis://:password@host. The userinfo (before '@') is what leaks secrets.
_URL_WITH_USERINFO = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://[^/\s@]*:[^/\s@]*@")

# Well-known secret token prefixes (GitHub, OpenAI/Stripe, AWS, Slack, Google...).
_SECRET_PREFIXES = (
    "sk-",
    "sk_live_",
    "sk_test_",
    "pk_live_",
    "rk_live_",
    "ghp_",
    "gho_",
    "ghu_",
    "ghs_",
    "ghr_",
    "github_pat_",
    "xox",  # Slack: xoxb-, xoxp-, xoxa-, xoxs-
    "AKIA",  # AWS access key id
    "ASIA",  # AWS temporary access key id
    "AIza",  # Google API key
    "ya29.",  # Google OAuth token
    "glpat-",  # GitLab personal access token
)


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _looks_like_secret_value(value: str) -> bool:
    """Heuristic: does *value* itself look like a credential?

    Deliberately conservative so ordinary config (paths, numbers, hostnames,
    booleans) is preserved. Flags: URLs/DSNs with inline userinfo, known secret
    prefixes, and long high-entropy opaque strings.
    """
    v = value.strip().strip("\"'")
    if not v:
        return False
    # Credentials embedded in a connection string / URL.
    if _URL_WITH_USERINFO.search(v):
        return True
    # Known provider token prefixes.
    if v.startswith(_SECRET_PREFIXES):
        return True
    # High-entropy opaque token: long, no whitespace, restricted alphabet, and
    # high per-character entropy. Excludes readable text (spaces / low entropy)
    # and short values to avoid over-redacting normal config.
    return (
        len(v) >= 24
        and not any(c.isspace() for c in v)
        and re.fullmatch(r"[A-Za-z0-9+/=_\-.]+", v) is not None
        and sum(c.isalpha() for c in v) > 0
        and sum(c.isdigit() for c in v) > 0
        and _shannon_entropy(v) >= 3.5
    )


# Redaction fires on either a sensitive key name OR a value that looks like a
# credential. The value check catches leaks the key-name check misses, e.g.
# DATABASE_URL=postgres://user:pass@host or CREDENTIAL=ghp_xxxx.
def _is_sensitive(key: str) -> bool:
    lower = key.lower()
    return any(kw in lower for kw in _SENSITIVE_KEYWORDS)


def _redact(key: str, value: str) -> str:
    if _is_sensitive(key) or _looks_like_secret_value(value):
        return "***"
    return value


def _parse_env(content: str) -> list[tuple[str, str]]:
    pairs = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            k, v = key.strip(), value.strip()
            if not k:
                continue
            pairs.append((k, v))
    return pairs


def _parse_ini(content: str) -> dict[str, list[tuple[str, str]]]:
    parser = configparser.ConfigParser()
    parser.read_string(content)
    sections: dict[str, list[tuple[str, str]]] = {}
    for section in parser.sections():
        sections[section] = [(k.strip(), v.strip()) for k, v in parser.items(section) if k.strip()]
    return sections


def _flatten_toml(data: dict, prefix: str = "") -> list[tuple[str, str]]:
    pairs = []
    for k, v in data.items():
        full_key = f"{prefix}.{k.strip()}" if prefix else k.strip()
        if not full_key:
            continue
        if isinstance(v, dict):
            pairs.extend(_flatten_toml(v, full_key))
        else:
            pairs.append((full_key, str(v).strip() if v is not None else ""))
    return pairs


class ConfigLoader:
    """Load .env, .ini, .cfg, or .toml config files into Documents.

    Sensitive keys (password, secret, token, api_key, key, auth) are redacted.
    Values that *look* like credentials are also redacted regardless of key
    name: connection strings/DSNs with inline userinfo (``scheme://user:pass@``),
    known token prefixes (``sk-``, ``ghp_``, ``AKIA``...), and long high-entropy
    opaque strings.
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def load(self) -> list[Document]:
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"Config file not found: {self._path}")

        basename = os.path.basename(self._path).lower()
        ext = os.path.splitext(self._path)[1].lower()
        # dotfiles like ".env" have no extension; treat the whole filename as the ext.
        # Also handles ".env.local", ".env.staging", etc.
        if not ext or basename.startswith(".env"):
            ext = ".env" if basename.startswith(".env") else basename
            if ext and not ext.startswith("."):
                ext = f".{ext}"
        if ext not in _SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported config file type: {ext!r}")

        with open(self._path, encoding="utf-8") as f:
            content = f.read()

        if ext == ".env":
            return self._load_env(content)
        if ext in {".ini", ".cfg"}:
            return self._load_ini(content)
        return self._load_toml(content)

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    # ------------------------------------------------------------------
    # Private parsers
    # ------------------------------------------------------------------

    def _load_env(self, content: str) -> list[Document]:
        pairs = _parse_env(content)
        lines = [f"{k}: {_redact(k, v)}" for k, v in pairs]
        text = "\n".join(lines)
        return [Document(text=text, metadata={"source": self._path, "type": "env"})]

    def _load_ini(self, content: str) -> list[Document]:
        sections = _parse_ini(content)
        if not sections:
            return [Document(text="", metadata={"source": self._path, "type": "ini"})]
        docs = []
        for section, pairs in sections.items():
            lines = [f"[{section}]"] + [f"{k}: {_redact(k, v)}" for k, v in pairs]
            docs.append(
                Document(
                    text="\n".join(lines),
                    metadata={"source": self._path, "type": "ini", "section": section},
                )
            )
        return docs

    def _load_toml(self, content: str) -> list[Document]:
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                raise ImportError(
                    "TOML loading requires Python 3.11+ or: pip install tomli"
                ) from None

        data = tomllib.loads(content)
        if not isinstance(data, dict):
            return [Document(text="", metadata={"source": self._path, "type": "toml"})]
        pairs = _flatten_toml(data)
        lines = [f"{k}: {_redact(k, v)}" for k, v in pairs]
        text = "\n".join(lines)
        return [Document(text=text, metadata={"source": self._path, "type": "toml"})]
