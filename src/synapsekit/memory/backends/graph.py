from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ...retrieval.property_graph import (
    NetworkXPropertyGraphBackend,
    PropertyGraphBackend,
    PropertyGraphEdge,
    PropertyGraphNode,
)
from ..base import BaseMemoryBackend, MemoryRecord, MemoryType


class GraphMemoryBackend(BaseMemoryBackend):
    """AgentMemory backend that stores memories as property-graph nodes."""

    def __init__(self, store: PropertyGraphBackend | None = None) -> None:
        self.graph_store = store or NetworkXPropertyGraphBackend()
        self._records: dict[str, dict[str, MemoryRecord]] = {}

    async def store(self, record: MemoryRecord) -> None:
        self._records.setdefault(record.agent_id, {})[record.id] = record
        self.graph_store.upsert_node(
            PropertyGraphNode(
                id=self._node_id(record),
                label=record.content,
                type=f"memory_{record.memory_type}",
                properties=self._record_properties(record),
            )
        )
        for related_id in self._related_ids(record.metadata):
            if related_id == record.id:
                continue
            self.graph_store.upsert_edge(
                PropertyGraphEdge(
                    source=self._node_id(record),
                    target=f"memory:{record.agent_id}:{related_id}",
                    relation=str(record.metadata.get("relation_type", "related_to")),
                    properties={
                        "weight": float(record.metadata.get("weight", 1.0)),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "relation_type": str(record.metadata.get("relation_type", "related_to")),
                        "agent_id": record.agent_id,
                    },
                )
            )

    async def fetch(
        self,
        agent_id: str,
        memory_type: MemoryType | None = None,
        *,
        include_expired: bool = False,
    ) -> list[MemoryRecord]:
        bucket = self._records.get(agent_id, {})
        now = datetime.now(timezone.utc)
        records: list[MemoryRecord] = []
        for record in bucket.values():
            if memory_type is not None and record.memory_type != memory_type:
                continue
            if not include_expired and record.is_expired(now):
                continue
            records.append(record)
        records.sort(key=lambda record: record.created_at)
        return records

    async def touch(
        self,
        agent_id: str,
        record_id: str,
        *,
        accessed_at: datetime | None = None,
    ) -> None:
        record = self._records.get(agent_id, {}).get(record_id)
        if record is None:
            return
        record.accessed_at = accessed_at or datetime.now(timezone.utc)
        record.access_count += 1
        node = self.graph_store.get_node(self._node_id(record))
        if node is not None:
            node.properties["accessed_at"] = record.accessed_at.isoformat()
            node.properties["access_count"] = record.access_count
            self.graph_store.upsert_node(node)

    async def delete(self, agent_id: str, record_id: str) -> bool:
        bucket = self._records.get(agent_id)
        if bucket is None:
            return False
        record = bucket.pop(record_id, None)
        if record is None:
            return False
        self.graph_store.remove_node(self._node_id(record))
        return True

    async def clear(self, agent_id: str, memory_type: MemoryType | None = None) -> int:
        bucket = self._records.get(agent_id)
        if bucket is None:
            return 0
        if memory_type is None:
            removed = len(bucket)
            bucket.clear()
            return removed
        to_remove = [
            record_id for record_id, record in bucket.items() if record.memory_type == memory_type
        ]
        for record_id in to_remove:
            bucket.pop(record_id, None)
        return len(to_remove)

    async def count(self, agent_id: str, memory_type: MemoryType | None = None) -> int:
        bucket = self._records.get(agent_id, {})
        if memory_type is None:
            return len(bucket)
        return sum(1 for record in bucket.values() if record.memory_type == memory_type)

    async def prune_expired(self, *, now: datetime | None = None) -> int:
        now_utc = now or datetime.now(timezone.utc)
        removed = 0
        for agent_id, bucket in list(self._records.items()):
            expired = [
                record_id for record_id, record in bucket.items() if record.is_expired(now_utc)
            ]
            for record_id in expired:
                bucket.pop(record_id, None)
                removed += 1
            if not bucket:
                self._records.pop(agent_id, None)
        return removed

    @staticmethod
    def _node_id(record: MemoryRecord) -> str:
        return f"memory:{record.agent_id}:{record.id}"

    @staticmethod
    def _related_ids(metadata: dict[str, Any]) -> list[str]:
        related = metadata.get("related_to") or metadata.get("related_ids") or []
        if isinstance(related, str):
            return [related]
        if isinstance(related, list | tuple | set):
            return [str(item) for item in related]
        return []

    @staticmethod
    def _record_properties(record: MemoryRecord) -> dict[str, Any]:
        return {
            **record.metadata,
            "record_id": record.id,
            "agent_id": record.agent_id,
            "memory_type": record.memory_type,
            "created_at": record.created_at.isoformat(),
            "accessed_at": record.accessed_at.isoformat(),
            "access_count": record.access_count,
            "ttl_days": record.ttl_days,
        }


GraphAgentMemory = GraphMemoryBackend
