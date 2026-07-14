"""Cross-project entity resolution and duplication detection."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..loaders.base import Document
from ..retrieval.world_model import EntityMention, EntityResolver, WorldModelNode

_CAMEL_SPLIT_RE = re.compile(r"(?<!^)(?=[A-Z])")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class CrossProjectEntityResolver(EntityResolver):
    """Resolve entities across repositories with conservative aliases."""

    def resolve(self, entity: EntityMention, nodes: dict[str, WorldModelNode]) -> str | None:
        direct = super().resolve(entity, nodes)
        if direct is not None:
            return direct

        candidate_keys = self._entity_keys(entity.name)
        for node_id, node in nodes.items():
            node_keys = self._entity_keys(node.name)
            if candidate_keys & node_keys:
                return node_id
            for alias in node.aliases:
                if candidate_keys & self._entity_keys(alias):
                    return node_id
        return None

    def remember(self, node: WorldModelNode) -> None:
        """Remember names, aliases, and cross-project concept aliases."""

        for alias in self._concept_aliases(node.name):
            node.aliases.add(alias)
        super().remember(node)

    @classmethod
    def _entity_keys(cls, value: str) -> set[str]:
        return {
            self_key for alias in cls._concept_aliases(value) if (self_key := cls._normalize(alias))
        }

    @staticmethod
    def _concept_aliases(value: str) -> set[str]:
        expanded = _CAMEL_SPLIT_RE.sub(" ", value).casefold()
        tokens = _TOKEN_RE.findall(expanded)
        aliases = {" ".join(tokens)} if tokens else set()
        suffixes = {
            "api",
            "class",
            "component",
            "handler",
            "module",
            "package",
            "repo",
            "service",
            "system",
        }
        trimmed = [token for token in tokens if token not in suffixes]
        if trimmed and trimmed != tokens:
            aliases.add(" ".join(trimmed))
        if len(trimmed) > 1:
            aliases.add(trimmed[0])
        return {alias for alias in aliases if len(alias) >= 3}


@dataclass(frozen=True)
class DuplicationMatch:
    """A likely duplicate concept or snippet across two mesh documents."""

    left_path: str
    right_path: str
    score: float
    shared_terms: tuple[str, ...]
    left_line_start: int | None = None
    right_line_start: int | None = None


class DuplicationDetector:
    """Detect repeated concepts across mesh chunks with token fingerprints."""

    def __init__(self, min_score: float = 0.72, min_terms: int = 4) -> None:
        self.min_score = min_score
        self.min_terms = min_terms

    def find(self, docs: list[Document], *, limit: int = 20) -> list[DuplicationMatch]:
        """Return likely duplicate chunk pairs."""

        fingerprints = [(doc, self._fingerprint(doc.text)) for doc in docs]
        matches: list[DuplicationMatch] = []
        for index, (left, left_terms) in enumerate(fingerprints):
            if len(left_terms) < self.min_terms:
                continue
            for right, right_terms in fingerprints[index + 1 :]:
                if len(right_terms) < self.min_terms:
                    continue
                if left.metadata.get("path") == right.metadata.get("path"):
                    continue
                shared = left_terms & right_terms
                union = left_terms | right_terms
                score = len(shared) / len(union) if union else 0.0
                if score < self.min_score:
                    continue
                matches.append(
                    DuplicationMatch(
                        left_path=str(left.metadata.get("path") or left.metadata.get("source")),
                        right_path=str(right.metadata.get("path") or right.metadata.get("source")),
                        score=score,
                        shared_terms=tuple(sorted(shared)[:16]),
                        left_line_start=_optional_int(left.metadata.get("line_start")),
                        right_line_start=_optional_int(right.metadata.get("line_start")),
                    )
                )

        return sorted(matches, key=lambda match: match.score, reverse=True)[:limit]

    @staticmethod
    def _fingerprint(text: str) -> set[str]:
        tokens = _TOKEN_RE.findall(text.casefold())
        stop = {
            "and",
            "for",
            "from",
            "have",
            "that",
            "the",
            "this",
            "with",
            "you",
            "your",
        }
        return {token for token in tokens if len(token) >= 4 and token not in stop}


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None
