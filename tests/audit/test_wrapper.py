"""VerifiableAgent — wraps an agent's methods without touching its implementation."""

from __future__ import annotations

import pytest

from synapsekit.audit import AuditTracer, EventKind, VerifiableAgent
from synapsekit.audit.wrapper import audited, infer_kind_pair


class DummyAgent:
    def __init__(self):
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        self.calls += 1
        return f"response to {prompt}"

    def retrieve(self, query: str) -> list[str]:
        return [f"doc-for-{query}"]

    async def failing_tool(self, x: int) -> int:
        raise ValueError("boom")


class TestInferKindPair:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("generate", (EventKind.LLM_CALL, EventKind.LLM_RESPONSE)),
            ("chat_complete", (EventKind.LLM_CALL, EventKind.LLM_RESPONSE)),
            ("retrieve_docs", (EventKind.RETRIEVAL, None)),
            ("recall_memory", (EventKind.MEMORY_READ, None)),
            ("save_memory", (EventKind.MEMORY_WRITE, None)),
            ("decide_next_step", (EventKind.DECISION, None)),
            ("run_tool", (EventKind.TOOL_CALL, EventKind.TOOL_RESULT)),
            ("frobnicate", (EventKind.SYSTEM_EVENT, None)),
        ],
    )
    def test_keyword_based_inference(self, name, expected):
        assert infer_kind_pair(name) == expected


class TestVerifiableAgent:
    @pytest.mark.asyncio
    async def test_paired_kind_emits_a_call_and_a_response_record(self):
        # "generate" -> (LLM_CALL, LLM_RESPONSE): a paired kind emits two
        # linked records, not one.
        tracer = AuditTracer()
        agent = VerifiableAgent(DummyAgent(), tracer=tracer)
        result = await agent.generate("hello")
        assert result == "response to hello"

        records = tracer.records
        assert len(records) == 2
        call, response = records
        assert call.kind == EventKind.LLM_CALL.value
        assert "output" not in call.payload
        assert response.kind == EventKind.LLM_RESPONSE.value
        assert response.payload["output"] == "response to hello"
        assert response.payload["status"] == "ok"
        assert response.parent_id == call.event_id

    def test_single_kind_emits_one_record(self):
        # "retrieve" -> RETRIEVAL has no paired response kind.
        tracer = AuditTracer()
        agent = VerifiableAgent(DummyAgent(), tracer=tracer)
        result = agent.retrieve("france")
        assert result == ["doc-for-france"]
        records = tracer.records
        assert len(records) == 1
        assert records[0].kind == EventKind.RETRIEVAL.value
        # Frozen payload: lists become tuples (see types.deep_freeze).
        assert records[0].payload["output"] == ("doc-for-france",)

    @pytest.mark.asyncio
    async def test_exceptions_are_recorded_as_error_and_reraised(self):
        # "failing_tool" -> paired (TOOL_CALL, TOOL_RESULT); an exception
        # replaces the result record's kind with ERROR.
        tracer = AuditTracer()
        agent = VerifiableAgent(DummyAgent(), tracer=tracer)
        with pytest.raises(ValueError, match="boom"):
            await agent.failing_tool(1)
        records = tracer.records
        assert len(records) == 2
        call, error = records
        assert call.kind == EventKind.TOOL_CALL.value
        assert error.kind == EventKind.ERROR.value
        assert error.payload["status"] == "error"
        assert error.payload["error_type"] == "ValueError"
        assert error.parent_id == call.event_id

    def test_non_callable_attributes_pass_through(self):
        tracer = AuditTracer()
        agent = VerifiableAgent(DummyAgent(), tracer=tracer)
        assert agent.calls == 0

    def test_uses_the_tracer_instance_passed_in_even_when_empty(self):
        # Regression: an empty AuditTracer defines __len__ == 0, which is
        # falsy — `tracer or AuditTracer()` would silently substitute a
        # fresh tracer instead of using the one the caller passed in.
        tracer = AuditTracer()
        agent = VerifiableAgent(DummyAgent(), tracer=tracer)
        assert agent.tracer is tracer

    def test_wrapped_property_exposes_the_underlying_agent(self):
        inner = DummyAgent()
        agent = VerifiableAgent(inner, tracer=AuditTracer())
        assert agent.wrapped is inner

    def test_pii_is_redacted_before_recording(self):
        from synapsekit.audit.redact import PIIRedactor

        tracer = AuditTracer()
        agent = VerifiableAgent(DummyAgent(), tracer=tracer, redactor=PIIRedactor())
        agent.retrieve("contact bob@example.com")
        payload = tracer.records[0].payload
        assert "bob@example.com" not in str(payload)

    def test_default_actor_names_the_wrapped_agent_type(self):
        tracer = AuditTracer()
        agent = VerifiableAgent(DummyAgent(), tracer=tracer)
        agent.retrieve("q")
        assert tracer.records[0].actor == "agent:DummyAgent"

    def test_custom_actor_is_used(self):
        tracer = AuditTracer()
        agent = VerifiableAgent(DummyAgent(), tracer=tracer, actor="user:alice")
        agent.retrieve("q")
        assert tracer.records[0].actor == "user:alice"


class _Retriever:
    async def search(self, query: str) -> list[str]:
        return [f"doc-for-{query}"]


class _AgentWithSubComponent:
    def __init__(self) -> None:
        self.retriever = _Retriever()

    async def run(self, query: str) -> str:
        # An internal self-call — NOT made through the outer proxy.
        docs = await self.retriever.search(query)
        return f"answer using {docs}"


class TestInstrumentAttrs:
    """Closes the gap where an agent's own internal calls (not made
    through the outer proxy) were invisible to the tracer."""

    @pytest.mark.asyncio
    async def test_internal_self_call_to_an_instrumented_sub_attr_is_captured(self):
        tracer = AuditTracer()
        agent = VerifiableAgent(
            _AgentWithSubComponent(), tracer=tracer, instrument_attrs=["retriever"]
        )

        result = await agent.run("hello")

        assert result == "answer using ['doc-for-hello']"
        # "run" -> paired (TOOL_CALL, TOOL_RESULT); "search" -> RETRIEVAL (single).
        kinds = sorted(r.kind for r in tracer.records)
        assert kinds == [
            EventKind.RETRIEVAL.value,
            EventKind.TOOL_CALL.value,
            EventKind.TOOL_RESULT.value,
        ]

    @pytest.mark.asyncio
    async def test_captured_internal_call_has_the_outer_call_as_its_parent(self):
        tracer = AuditTracer()
        agent = VerifiableAgent(
            _AgentWithSubComponent(), tracer=tracer, instrument_attrs=["retriever"]
        )
        await agent.run("hello")

        call_record = next(r for r in tracer.records if r.kind == EventKind.TOOL_CALL.value)
        search_record = next(r for r in tracer.records if r.kind == EventKind.RETRIEVAL.value)
        result_record = next(r for r in tracer.records if r.kind == EventKind.TOOL_RESULT.value)
        assert search_record.parent_id == call_record.event_id
        assert result_record.parent_id == call_record.event_id

    @pytest.mark.asyncio
    async def test_without_instrument_attrs_the_internal_call_is_invisible(self):
        tracer = AuditTracer()
        agent = VerifiableAgent(_AgentWithSubComponent(), tracer=tracer)
        await agent.run("hello")

        # Documents the known limitation: only the outermost proxied call
        # is captured (as its call/result pair) unless the sub-attribute
        # is explicitly instrumented — the internal `search` call never
        # appears.
        assert len(tracer.records) == 2
        assert all(r.payload["method"] == "run" for r in tracer.records)
        assert {r.kind for r in tracer.records} == {
            EventKind.TOOL_CALL.value,
            EventKind.TOOL_RESULT.value,
        }

    def test_instrumenting_a_missing_attr_is_a_no_op(self):
        tracer = AuditTracer()
        # Should not raise even though DummyAgent has no 'nonexistent' attr.
        VerifiableAgent(DummyAgent(), tracer=tracer, instrument_attrs=["nonexistent"])


class TestAuditedDecorator:
    @pytest.mark.asyncio
    async def test_decorates_a_standalone_async_tool(self):
        tracer = AuditTracer()

        @audited(EventKind.TOOL_CALL, tracer=tracer)
        async def search(query: str) -> str:
            return f"results for {query}"

        result = await search("weather")
        assert result == "results for weather"
        assert len(tracer.records) == 1
        assert tracer.records[0].kind == EventKind.TOOL_CALL.value

    def test_decorates_a_standalone_sync_tool(self):
        tracer = AuditTracer()

        @audited(EventKind.TOOL_CALL, tracer=tracer)
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5
        assert tracer.records[0].payload["output"] == 5

    @pytest.mark.asyncio
    async def test_nested_audited_calls_get_correct_parent_id(self):
        tracer = AuditTracer()

        @audited(EventKind.TOOL_CALL, tracer=tracer, name="inner")
        async def inner() -> str:
            return "done"

        @audited(EventKind.DECISION, tracer=tracer, name="outer")
        async def outer() -> str:
            return await inner()

        await outer()
        records = {r.payload["method"]: r for r in tracer.records}
        assert records["inner"].parent_id == records["outer"].event_id

    @pytest.mark.asyncio
    async def test_exception_is_recorded_with_error_kind_regardless_of_declared_kind(self):
        tracer = AuditTracer()

        @audited(EventKind.TOOL_CALL, tracer=tracer)
        async def boom() -> None:
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            await boom()
        assert tracer.records[0].kind == EventKind.ERROR.value

    def test_default_actor_is_system(self):
        tracer = AuditTracer()

        @audited(EventKind.TOOL_CALL, tracer=tracer)
        def noop() -> None:
            return None

        noop()
        assert tracer.records[0].actor == "system"
