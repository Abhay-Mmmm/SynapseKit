"""Regression tests for #785 / #789 — CassandraVectorStore CQL building.

The CQL search query used f-string interpolation of top_k directly into the LIMIT
clause. The fix validates/casts top_k to a positive int and passes it as a bound
parameter (``%s``); only the query vector stays interpolated because CQL's
``ORDER BY ... ANN OF`` clause requires a vector literal, not a bind marker.

We do not need astrapy or cassandra-driver installed: we build the store via
``object.__new__`` and drive ``_cass_search_sync`` with a hand-written fake session
that captures the CQL string and bound parameters.
"""

from __future__ import annotations

import json

import pytest

from synapsekit.retrieval.cassandra_vector import CassandraVectorStore


class FakeRow:
    def __init__(self, text: str, metadata: str | None) -> None:
        self.text = text
        self.metadata = metadata


class FakeSession:
    def __init__(self, rows: list[FakeRow] | None = None) -> None:
        self.rows = rows or []
        self.executed: list[tuple] = []

    def execute(self, cql, params=None):
        self.executed.append((cql, params))
        return self.rows


def _make_store(session: FakeSession) -> CassandraVectorStore:
    store = object.__new__(CassandraVectorStore)
    store._keyspace = "ks"
    store._table_name = "vec"
    store._session = session
    return store


def test_top_k_passed_as_bound_parameter_not_interpolated():
    session = FakeSession()
    store = _make_store(session)

    store._cass_search_sync([0.1, 0.2, 0.3], top_k=7, metadata_filter=None)

    cql, params = session.executed[0]
    assert "LIMIT %s" in cql
    assert "LIMIT 7" not in cql  # value must not be inlined
    assert params == (7,)


def test_vector_literal_is_built_from_floats():
    session = FakeSession()
    store = _make_store(session)

    store._cass_search_sync([1.0, 2.0], top_k=3, metadata_filter=None)

    cql, _ = session.executed[0]
    assert "ANN OF [1.0,2.0]" in cql


def test_non_positive_top_k_rejected():
    store = _make_store(FakeSession())
    with pytest.raises(ValueError):
        store._cass_search_sync([0.1], top_k=0, metadata_filter=None)
    with pytest.raises(ValueError):
        store._cass_search_sync([0.1], top_k=-4, metadata_filter=None)


def test_string_top_k_injection_payload_rejected():
    store = _make_store(FakeSession())
    with pytest.raises(ValueError):
        store._cass_search_sync([0.1], top_k="5; DROP TABLE vec", metadata_filter=None)


def test_metadata_filter_still_applied_to_rows():
    rows = [
        FakeRow("keep", json.dumps({"lang": "en"})),
        FakeRow("drop", json.dumps({"lang": "fr"})),
    ]
    store = _make_store(FakeSession(rows))

    results = store._cass_search_sync([0.1], top_k=5, metadata_filter={"lang": "en"})

    assert [r["text"] for r in results] == ["keep"]


def test_cassandra_extra_declares_astrapy():
    """#789: the [cassandra] extra must include astrapy so the astra path's
    ImportError message ('pip install synapsekit[cassandra]') is accurate."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    cassandra_line = next(
        line for line in pyproject.splitlines() if line.startswith("cassandra = [")
    )
    assert "astrapy" in cassandra_line
