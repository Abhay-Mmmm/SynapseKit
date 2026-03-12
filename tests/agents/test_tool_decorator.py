"""Tests for @tool decorator."""

import pytest

from synapsekit.agents.base import BaseTool, ToolResult
from synapsekit.agents.tool_decorator import tool


class TestToolDecorator:
    def test_basic_sync_tool(self):
        @tool(name="add", description="Add two numbers")
        def add(a: int, b: int) -> str:
            return str(a + b)

        assert isinstance(add, BaseTool)
        assert add.name == "add"
        assert add.description == "Add two numbers"

    def test_infers_name_and_description(self):
        @tool()
        def multiply(a: int, b: int) -> str:
            """Multiply two numbers."""
            return str(a * b)

        assert multiply.name == "multiply"
        assert multiply.description == "Multiply two numbers."

    @pytest.mark.asyncio
    async def test_sync_tool_run(self):
        @tool(name="greet", description="Greet someone")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        result = await greet.run(name="World")
        assert isinstance(result, ToolResult)
        assert result.output == "Hello, World!"
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_async_tool_run(self):
        @tool(name="async_add", description="Async add")
        async def async_add(a: int, b: int) -> str:
            return str(a + b)

        result = await async_add.run(a=3, b=4)
        assert result.output == "7"

    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        @tool(name="fail", description="Always fails")
        def fail() -> str:
            raise ValueError("boom")

        result = await fail.run()
        assert result.is_error
        assert "boom" in result.error

    def test_schema_generation(self):
        @tool(name="search", description="Search for something")
        def search(query: str, limit: int = 10) -> str:
            return ""

        schema = search.schema()
        assert schema["function"]["name"] == "search"
        params = schema["function"]["parameters"]
        assert "query" in params["properties"]
        assert params["properties"]["query"]["type"] == "string"
        assert params["properties"]["limit"]["type"] == "integer"
        assert "query" in params["required"]
        assert "limit" not in params["required"]

    def test_tool_works_with_registry(self):
        from synapsekit.agents.registry import ToolRegistry

        @tool(name="calc", description="Calculate")
        def calc(expr: str) -> str:
            return str(eval(expr))

        registry = ToolRegistry([calc])
        assert registry.get("calc") is calc
        schemas = registry.schemas()
        assert len(schemas) == 1
