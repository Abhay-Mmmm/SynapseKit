from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Result returned by any tool execution."""

    output: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def __str__(self) -> str:
        return self.error if self.error is not None else self.output


class BaseTool(ABC):
    """Abstract base class for all agent tools."""

    name: str
    description: str

    # JSON Schema for the tool's input parameters.
    #
    # Exposed as a property (not a bare class attribute) so the *default* is a
    # fresh, JSON-serialisable dict — never a ``dataclasses.Field`` object (the
    # #799 crash) and never one mutable dict shared across instances.
    #
    # Subclasses may still override in either idiomatic way:
    #   * a class-level ``parameters = {...}`` literal (shadows this property), or
    #   * assignment in ``__init__`` (``self.parameters = ...``) — the setter
    #     below stores it as a per-instance override.
    @property
    def parameters(self) -> dict[str, Any]:
        override = self.__dict__.get("_parameters_override")
        if override is not None:
            return override
        return {"type": "object", "properties": {}}

    @parameters.setter
    def parameters(self, value: dict[str, Any]) -> None:
        self.__dict__["_parameters_override"] = value

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool. kwargs come from the parsed Action Input."""
        ...

    def schema(self) -> dict:
        """OpenAI-compatible function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def anthropic_schema(self) -> dict:
        """Anthropic-compatible tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
