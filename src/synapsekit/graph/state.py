"""Graph state definitions and typed state with reducers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Sentinel: a node's condition_fn returns END to signal graph termination.
END = "__end__"

# Type alias — graphs pass around plain dicts.
GraphState = dict[str, Any]


@dataclass
class StateField:
    """A typed state field with an optional reducer for parallel merge.

    The reducer function receives ``(current_value, new_value)`` and returns
    the merged result.  Without a reducer, the default behavior is
    ``dict.update()`` — last write wins.

    Usage::

        from synapsekit.graph.state import StateField, TypedState

        schema = TypedState(
            fields={
                "messages": StateField(default=list, reducer=lambda cur, new: cur + new),
                "count": StateField(default=int, reducer=lambda cur, new: cur + new),
                "result": StateField(default=str),  # last-write-wins
            }
        )
        graph = StateGraph(state_schema=schema)
    """

    default: Callable[[], Any] | Any = None
    reducer: Callable[[Any, Any], Any] | None = None


@dataclass
class TypedState:
    """Schema for typed graph state with per-field reducers.

    Fields without a reducer use last-write-wins semantics.
    Fields with a reducer use ``reducer(current, new)`` for safe parallel merge.
    """

    fields: dict[str, StateField] = field(default_factory=dict)

    def initial_state(self) -> dict[str, Any]:
        """Create the initial state dict from field defaults."""
        state: dict[str, Any] = {}
        for name, sf in self.fields.items():
            if callable(sf.default):
                state[name] = sf.default()
            else:
                state[name] = sf.default
        return state

    def merge(self, current: dict[str, Any], partial: dict[str, Any]) -> None:
        """Merge a partial state update into the current state using reducers.

        Modifies ``current`` in-place.
        """
        for key, value in partial.items():
            sf = self.fields.get(key)
            if sf is not None and sf.reducer is not None:
                default = sf.default() if callable(sf.default) else sf.default
                current[key] = sf.reducer(current.get(key, default), value)
            else:
                current[key] = value
