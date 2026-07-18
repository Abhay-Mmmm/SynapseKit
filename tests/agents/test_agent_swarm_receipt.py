"""Spec test for the replayable role-allocation receipt requested on #734 by clementineCU.

Every auction should emit a receipt that can be replayed without trusting the
final answer: task_id, candidate agents, bid inputs each agent was allowed to
see, cost/quality prior version, bid value, selected role, rejected roles,
budget consumed, outcome score source, and the rule that updates future bids.

This currently fails — AgentSwarm.trace records bids/winners/reward but not
the reputation prior used to score them, the budget allocated, or whether the
outcome score was externally supplied vs. self-reported by the winning agent.
Remove the xfail once the receipt carries these fields.
"""

import pytest

from synapsekit import AgentSwarm, Bid, MarketPolicy
from synapsekit.agents import AgentMetadata, InMemoryAgentRegistry


class MarketClient:
    def __init__(self, agent_id: str, *, cost: float, quality: float) -> None:
        self.agent_id = agent_id
        self.cost = cost
        self.quality = quality

    def bid(self, task: str, **kwargs):
        return Bid(
            agent_id=self.agent_id,
            estimated_cost=self.cost,
            estimated_quality=self.quality,
            confidence=0.9,
            task_category=kwargs["task_category"],
        )

    async def run(self, prompt: str, **kwargs):
        # No actual_cost/actual_quality/reward in the output — forces AgentSwarm
        # to fall back to self-reported bid values when settling the outcome.
        return {"agent_id": self.agent_id, "prompt": prompt}


@pytest.mark.xfail(
    reason="#734: auction receipts don't yet carry task_id, reputation prior, "
    "budget consumed, or outcome-score provenance — see clementineCU's comment",
    strict=True,
)
async def test_auction_receipt_is_replayable_without_trusting_final_answer():
    registry = InMemoryAgentRegistry()
    swarm = AgentSwarm(
        market=MarketPolicy(budget_per_task=100, seed=42, exploration_rate=0),
        registry=registry,
    )
    swarm.register_agent(
        AgentMetadata(id="researcher", model="mock", cost_multiplier=30.0, capacity=2),
        client=MarketClient("researcher", cost=30.0, quality=0.96),
    )
    swarm.register_agent(
        AgentMetadata(id="summarizer", model="mock", cost_multiplier=8.0, capacity=2),
        client=MarketClient("summarizer", cost=8.0, quality=0.74),
    )

    await swarm.execute("Write a market brief", task_category="research")

    receipt = swarm.trace[-1]

    # Stable identifier for the task, independent of its free-text prompt.
    assert "task_id" in receipt

    # Candidate agents + the exact bid inputs each agent submitted.
    assert {"researcher", "summarizer"} == {b["agent_id"] for b in receipt["bids"]}
    for bid in receipt["bids"]:
        assert "estimated_cost" in bid
        assert "estimated_quality" in bid

    # The reputation snapshot each bid was scored against, so a reviewer can
    # tell whether a win came from a real track record or a stale prior.
    for bid in receipt["bids"]:
        assert "reputation_prior" in bid
        assert "mean_quality" in bid["reputation_prior"]
        assert "version" in bid["reputation_prior"]

    # Selected vs. rejected roles must both be first-class, not derived.
    assert receipt["selected_roles"] == ["researcher"]
    assert receipt["rejected_roles"] == ["summarizer"]

    # Budget allocated vs. actually consumed by the winner(s).
    assert receipt["budget_allocated"] == 100
    assert "budget_consumed" in receipt

    # Where the outcome score came from: caller override, extracted from the
    # winning agent's own output, or a self-reported bid-value fallback. This
    # client returns neither actual_quality nor reward, so the source here
    # must be flagged as the self-reported fallback, not an external eval.
    assert receipt["outcome_score_source"] == "self_reported_bid_fallback"

    # The learning rule (and its version) that will use this outcome to
    # update future bids, so drift in the rule itself is auditable.
    assert receipt["learning_rule"]["name"] == "ema"
    assert receipt["learning_rule"]["learning_rate"] == 0.1
    assert "version" in receipt["learning_rule"]
