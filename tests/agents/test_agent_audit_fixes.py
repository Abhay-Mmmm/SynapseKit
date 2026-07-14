"""Regression tests for the agent audit fixes (issues #799-#805).

Each test fails against the pre-fix code and passes against the fix. No
MagicMock / Mock is used anywhere — real objects, hand-written fakes, and
``tmp_path`` provide all boundaries.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from synapsekit.agents.base import BaseTool, ToolResult
from synapsekit.agents.federation import (
    AgentFederation,
    AgentMetadata,
    InMemoryAgentRegistry,
    RoutingStrategy,
)
from synapsekit.agents.react import _parse_action
from synapsekit.agents.tools.browser import BrowserTool
from synapsekit.agents.tools.calculator import CalculatorTool
from synapsekit.agents.tools.file_read import FileReadTool
from synapsekit.agents.tools.file_write import FileWriteTool
from synapsekit.agents.tools.shell import ShellTool

# ---------------------------------------------------------------------------
# #799 — BaseTool.parameters must be a real, JSON-serialisable dict
# ---------------------------------------------------------------------------


class _ParamlessTool(BaseTool):
    """A tool that does NOT define ``parameters`` — exercises the default."""

    name = "paramless"
    description = "A tool with no custom parameters."

    async def run(self, **kwargs: object) -> ToolResult:
        return ToolResult(output="ok")


class TestParametersSerialisable:
    def test_default_parameters_is_a_dict(self) -> None:
        tool = _ParamlessTool()
        assert isinstance(tool.parameters, dict)
        assert tool.parameters == {"type": "object", "properties": {}}

    def test_schema_is_json_serialisable(self) -> None:
        # Pre-fix: parameters was a dataclasses.Field -> json.dumps raised.
        tool = _ParamlessTool()
        dumped = json.dumps(tool.schema())
        assert "paramless" in dumped

    def test_anthropic_schema_is_json_serialisable(self) -> None:
        tool = _ParamlessTool()
        dumped = json.dumps(tool.anthropic_schema())
        assert "paramless" in dumped

    def test_default_not_shared_across_instances(self) -> None:
        # Mutating one instance's default dict must not leak to another.
        a = _ParamlessTool()
        b = _ParamlessTool()
        a.parameters["properties"]["injected"] = {"type": "string"}
        assert "injected" not in b.parameters["properties"]

    def test_subclass_override_still_wins(self) -> None:
        # A subclass class-attr must shadow the default cleanly.
        calc = CalculatorTool()
        assert calc.parameters["required"] == ["expression"]
        json.dumps(calc.schema())  # must not raise


# ---------------------------------------------------------------------------
# #800 — Action Input regex must stop at the next section marker
# ---------------------------------------------------------------------------


class TestActionInputParsing:
    def test_multi_block_input_is_bounded(self) -> None:
        completion = (
            "Thought: I should search.\n"
            "Action: search\n"
            "Action Input: capital of France\n"
            "Observation: Paris\n"
            "Thought: Now I know.\n"
            "Final Answer: Paris"
        )
        action, action_input = _parse_action(completion)
        assert action == "search"
        # Pre-fix: DOTALL swallowed everything after "Action Input:".
        assert action_input == "capital of France"

    def test_hallucinated_observation_not_captured(self) -> None:
        completion = (
            "Action: calculator\n"
            "Action Input: 2 + 2\n"
            "Observation: 4\n"
            "Action: calculator\n"
            "Action Input: 3 + 3"
        )
        _, action_input = _parse_action(completion)
        assert action_input == "2 + 2"

    def test_single_block_still_works(self) -> None:
        completion = "Action: search\nAction Input: hello world"
        action, action_input = _parse_action(completion)
        assert action == "search"
        assert action_input == "hello world"

    def test_multiline_input_within_block_preserved(self) -> None:
        completion = (
            "Action: code\n"
            "Action Input: line1\nstill_input\n"
            "Final Answer: done"
        )
        _, action_input = _parse_action(completion)
        assert action_input == "line1\nstill_input"


# ---------------------------------------------------------------------------
# #801 — Calculator must reject RCE / DoS payloads via AST allowlist
# ---------------------------------------------------------------------------


class TestCalculatorSafety:
    async def test_subclasses_escape_rejected(self) -> None:
        tool = CalculatorTool()
        payload = "().__class__.__bases__[0].__subclasses__()"
        result = await tool.run(expression=payload)
        assert result.is_error
        assert "os" not in result.output

    async def test_dunder_attribute_access_rejected(self) -> None:
        tool = CalculatorTool()
        result = await tool.run(expression="(1).__class__")
        assert result.is_error

    async def test_import_style_name_rejected(self) -> None:
        tool = CalculatorTool()
        result = await tool.run(expression="__import__('os').system('echo pwned')")
        assert result.is_error

    async def test_huge_exponent_dos_rejected(self) -> None:
        tool = CalculatorTool()
        result = await tool.run(expression="9**9**9")
        assert result.is_error
        assert "exponent" in result.error.lower()

    async def test_subscripting_rejected(self) -> None:
        tool = CalculatorTool()
        result = await tool.run(expression="[1, 2, 3][0]")
        assert result.is_error

    async def test_non_whitelisted_call_rejected(self) -> None:
        tool = CalculatorTool()
        result = await tool.run(expression="open('x')")
        assert result.is_error

    async def test_basic_arithmetic_works(self) -> None:
        tool = CalculatorTool()
        result = await tool.run(expression="2 + 2")
        assert not result.is_error
        assert result.output == "4"

    async def test_math_functions_work(self) -> None:
        tool = CalculatorTool()
        result = await tool.run(expression="sqrt(16)")
        assert not result.is_error
        assert float(result.output) == 4.0

    async def test_constants_and_operators_work(self) -> None:
        tool = CalculatorTool()
        result = await tool.run(expression="round(pi * 2, 2)")
        assert not result.is_error
        assert result.output == "6.28"

    async def test_small_power_still_allowed(self) -> None:
        tool = CalculatorTool()
        result = await tool.run(expression="2 ** 10")
        assert not result.is_error
        assert result.output == "1024"


# ---------------------------------------------------------------------------
# #802 — Shell allowlist must not be bypassable via chained commands
# ---------------------------------------------------------------------------


class TestShellAllowlistBypass:
    async def test_chained_command_rejected(self) -> None:
        tool = ShellTool(allowed_commands=["echo"])
        result = await tool.run(command="echo hi & curl evil.example.com")
        assert result.is_error
        # Must be rejected as a policy violation, not executed.
        assert "metacharacter" in result.error or "not in the allowed" in result.error

    async def test_pipe_bypass_rejected(self) -> None:
        tool = ShellTool(allowed_commands=["echo"])
        result = await tool.run(command="echo hi | sh")
        assert result.is_error

    async def test_semicolon_bypass_rejected(self) -> None:
        tool = ShellTool(allowed_commands=["echo"])
        result = await tool.run(command="echo hi; whoami")
        assert result.is_error

    async def test_command_substitution_rejected(self) -> None:
        tool = ShellTool(allowed_commands=["echo"])
        result = await tool.run(command="echo $(whoami)")
        assert result.is_error

    async def test_allowed_simple_command_runs(self) -> None:
        tool = ShellTool(allowed_commands=["echo"])
        result = await tool.run(command="echo hello")
        assert not result.is_error
        assert "hello" in result.output

    async def test_disallowed_binary_rejected(self) -> None:
        tool = ShellTool(allowed_commands=["echo"])
        result = await tool.run(command="curl evil.example.com")
        assert result.is_error
        assert "not in the allowed" in result.error


# ---------------------------------------------------------------------------
# #803 — Browser must block SSRF to private / loopback / link-local IPs
# ---------------------------------------------------------------------------


class TestBrowserSSRFGuard:
    def test_cloud_metadata_ip_blocked(self) -> None:
        tool = BrowserTool()
        with pytest.raises(ValueError, match="private"):
            tool._validate_url("http://169.254.169.254/latest/meta-data/")

    def test_loopback_blocked(self) -> None:
        tool = BrowserTool()
        with pytest.raises(ValueError, match="private"):
            tool._validate_url("http://127.0.0.1:8080/admin")

    def test_private_10_range_blocked(self) -> None:
        tool = BrowserTool()
        with pytest.raises(ValueError, match="private"):
            tool._validate_url("http://10.0.0.5/")

    def test_ipv6_loopback_blocked(self) -> None:
        tool = BrowserTool()
        with pytest.raises(ValueError, match="private"):
            tool._validate_url("http://[::1]/")

    def test_localhost_hostname_blocked_via_resolution(self) -> None:
        # localhost resolves to a loopback address — must be rejected.
        tool = BrowserTool()
        with pytest.raises(ValueError):
            tool._validate_url("http://localhost/")

    def test_guard_applies_even_with_allowlist(self) -> None:
        # An allowlisted domain name pointing at a private literal IP: the
        # literal-IP form must still be blocked regardless of the allowlist.
        tool = BrowserTool(allowed_domains=["10.0.0.5"])
        with pytest.raises(ValueError, match="private"):
            tool._validate_url("http://10.0.0.5/")

    def test_opt_in_allows_private_ips(self) -> None:
        tool = BrowserTool(allow_private_ips=True)
        # Should not raise on the SSRF guard (allowlist is None -> allow-all).
        tool._validate_url("http://127.0.0.1/")

    def test_non_http_scheme_still_blocked(self) -> None:
        tool = BrowserTool()
        with pytest.raises(ValueError, match="scheme"):
            tool._validate_url("file:///etc/passwd")


# ---------------------------------------------------------------------------
# #804 — Federation must fail over to a healthy agent on crash / timeout
# ---------------------------------------------------------------------------


class _CrashingClient:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, prompt: str, **kwargs: object) -> object:
        self.calls += 1
        raise RuntimeError("agent crashed")


class _HangingClient:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, prompt: str, **kwargs: object) -> object:
        self.calls += 1
        await asyncio.sleep(60)  # would hang without wait_for
        return {"never": "returned"}


class _HealthyClient:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.calls = 0

    async def run(self, prompt: str, **kwargs: object) -> object:
        self.calls += 1
        return {"agent_id": self.agent_id, "prompt": prompt}


def _federation_with(bad_client: object, healthy_client: _HealthyClient) -> AgentFederation:
    registry = InMemoryAgentRegistry(stale_timeout=1_000_000)
    fed = AgentFederation(registry, default_strategy=RoutingStrategy.COST_AWARE)
    # Cost-aware ranks the cheaper (bad) agent first so failover is exercised.
    fed.register_agent(
        AgentMetadata(id="bad", model="m", tools=["t"], capacity=1, cost_multiplier=0.1),
        client=bad_client,
    )
    fed.register_agent(
        AgentMetadata(id="good", model="m", tools=["t"], capacity=1, cost_multiplier=0.5),
        client=healthy_client,
    )
    return fed


class TestFederationFailover:
    async def test_failover_on_crash(self) -> None:
        bad = _CrashingClient()
        good = _HealthyClient("good")
        fed = _federation_with(bad, good)
        result = await fed.run("hi", tools="t")
        assert result["agent_id"] == "good"
        assert bad.calls == 1
        assert good.calls == 1

    async def test_failover_on_timeout(self) -> None:
        bad = _HangingClient()
        good = _HealthyClient("good")
        fed = _federation_with(bad, good)
        result = await fed.run("hi", tools="t", timeout=0.1)
        assert result["agent_id"] == "good"
        assert bad.calls == 1
        assert good.calls == 1

    async def test_all_failing_raises(self) -> None:
        bad1 = _CrashingClient()
        bad2 = _CrashingClient()
        registry = InMemoryAgentRegistry(stale_timeout=1_000_000)
        fed = AgentFederation(registry)
        fed.register_agent(
            AgentMetadata(id="b1", model="m", tools=["t"], capacity=1), client=bad1
        )
        fed.register_agent(
            AgentMetadata(id="b2", model="m", tools=["t"], capacity=1), client=bad2
        )
        with pytest.raises(RuntimeError, match="candidate agents failed"):
            await fed.run("hi", tools="t")
        assert bad1.calls == 1
        assert bad2.calls == 1


# ---------------------------------------------------------------------------
# #805 — File tools must be non-blocking and enforce a byte cap
# ---------------------------------------------------------------------------


class TestFileToolSizeCaps:
    async def test_file_read_size_cap(self, tmp_path) -> None:
        big = tmp_path / "big.txt"
        big.write_text("x" * 5000)
        tool = FileReadTool(max_bytes=1000)
        result = await tool.run(path=str(big))
        assert result.is_error
        assert "too large" in result.error

    async def test_file_read_under_cap_works(self, tmp_path) -> None:
        f = tmp_path / "small.txt"
        f.write_text("hello")
        tool = FileReadTool(max_bytes=1000)
        result = await tool.run(path=str(f))
        assert not result.is_error
        assert result.output == "hello"

    async def test_file_write_size_cap(self, tmp_path) -> None:
        tool = FileWriteTool(base_dir=str(tmp_path), max_bytes=100)
        target = tmp_path / "out.txt"
        result = await tool.run(path=str(target), content="y" * 500)
        assert result.is_error
        assert "too large" in result.error
        assert not target.exists()

    async def test_file_write_under_cap_works(self, tmp_path) -> None:
        tool = FileWriteTool(base_dir=str(tmp_path), max_bytes=1000)
        target = tmp_path / "out.txt"
        result = await tool.run(path=str(target), content="ok")
        assert not result.is_error
        assert target.read_text() == "ok"

    async def test_read_write_roundtrip_offloads_to_thread(self, tmp_path) -> None:
        # A functional smoke test that the to_thread path returns correctly.
        target = tmp_path / "rt.txt"
        writer = FileWriteTool(base_dir=str(tmp_path))
        reader = FileReadTool(base_dir=str(tmp_path))
        await writer.run(path=str(target), content="roundtrip")
        result = await reader.run(path=str(target))
        assert result.output == "roundtrip"
