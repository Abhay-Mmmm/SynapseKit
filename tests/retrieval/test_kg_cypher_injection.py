"""Regression tests for #784 — Cypher injection via unchecked max_hops.

Neo4jStore.get_neighbors interpolates max_hops into the ``*1..N`` variable-length
pattern (Cypher requires an integer literal there, not a bound parameter). Old code
interpolated it raw, so a string like ``"1]->() DETACH DELETE (e) //"`` would be
injected. The fix casts to a bounded int.

No live Neo4j / neo4j driver is needed: we build the store via ``object.__new__``
and inject a hand-written fake driver that captures the compiled query text.
"""

from __future__ import annotations

import pytest

from synapsekit.retrieval.kg.backends import _MAX_HOPS, Neo4jStore


class FakeResult:
    def __iter__(self):
        return iter(())


class FakeSession:
    def __init__(self, recorder: dict) -> None:
        self._recorder = recorder

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self._recorder["query"] = query
        self._recorder["params"] = params
        return FakeResult()


class FakeDriver:
    def __init__(self) -> None:
        self.recorder: dict = {}

    def session(self):
        return FakeSession(self.recorder)


def _make_store() -> tuple[Neo4jStore, FakeDriver]:
    store = object.__new__(Neo4jStore)
    driver = FakeDriver()
    store._driver = driver
    return store, driver


def test_valid_max_hops_is_interpolated_safely():
    store, driver = _make_store()
    store.get_neighbors("Alice", max_hops=3)
    assert "*1..3]" in driver.recorder["query"]
    # entity is still passed as a bound parameter.
    assert driver.recorder["params"]["entity"] == "Alice"


def test_string_max_hops_with_injection_payload_is_rejected():
    store, _ = _make_store()
    with pytest.raises(ValueError):
        store.get_neighbors("Alice", max_hops="1]-() DETACH DELETE (e) //")  # type: ignore[arg-type]


def test_max_hops_zero_rejected():
    store, _ = _make_store()
    with pytest.raises(ValueError):
        store.get_neighbors("Alice", max_hops=0)


def test_max_hops_above_ceiling_rejected():
    store, _ = _make_store()
    with pytest.raises(ValueError):
        store.get_neighbors("Alice", max_hops=_MAX_HOPS + 1)


def test_float_max_hops_truncates_to_int():
    store, driver = _make_store()
    store.get_neighbors("Alice", max_hops=2.9)  # type: ignore[arg-type]
    assert "*1..2]" in driver.recorder["query"]
