"""Tests for pacman/agents/ — ghost, pacman, and keyboard agents."""
import pytest
from pacman.game import Agent
from pacman.agents.ghost_agents import GhostAgent, RandomGhost, DirectionalGhost
from pacman.agents.pacman_agents import LeftTurnAgent, GreedyAgent


# ── GhostAgent ───────────────────────────────────────────────────────────────

class TestGhostAgent:
    def test_init(self):
        agent = GhostAgent(1)
        assert agent.index == 1

    def test_get_distribution_raises(self):
        agent = GhostAgent(1)
        with pytest.raises(SystemExit):
            agent.get_distribution(None)


# ── RandomGhost ──────────────────────────────────────────────────────────────

class TestRandomGhost:
    def test_init(self):
        ghost = RandomGhost(1)
        assert ghost.index == 1

    def test_get_distribution(self):
        """RandomGhost should give uniform distribution over legal actions."""
        import pacman
        from pacman.layout import Layout

        lay = Layout([
            "%%%%%%",
            "% G  %",
            "%  P %",
            "%%%%%%",
        ])
        state = pacman.GameState()
        state.initialize(lay, 1)

        ghost = RandomGhost(1)
        dist = ghost.get_distribution(state)

        # Distribution should sum to approximately 1
        total = sum(dist.values())
        assert abs(total - 1.0) < 0.01

        # All legal ghost actions should have non-zero probability
        legal_actions = state.get_legal_actions(1)
        for action in legal_actions:
            assert dist[action] > 0


# ── DirectionalGhost ─────────────────────────────────────────────────────────

class TestDirectionalGhost:
    def test_init_default(self):
        ghost = DirectionalGhost(1)
        assert ghost.index == 1

    def test_init_custom(self):
        ghost = DirectionalGhost(2, prob_attack=0.5, prob_scared_flee=0.5)
        assert ghost.index == 2

    def test_get_distribution(self):
        """DirectionalGhost should have a valid probability distribution."""
        import pacman
        from pacman.layout import Layout

        lay = Layout([
            "%%%%%%",
            "% G  %",
            "%  P %",
            "%%%%%%",
        ])
        state = pacman.GameState()
        state.initialize(lay, 1)

        ghost = DirectionalGhost(1)
        dist = ghost.get_distribution(state)

        total = sum(dist.values())
        assert abs(total - 1.0) < 0.01


# ── LeftTurnAgent ────────────────────────────────────────────────────────────

class TestLeftTurnAgent:
    def test_init(self):
        agent = LeftTurnAgent()
        assert isinstance(agent, Agent)

    def test_get_action(self):
        """LeftTurnAgent should pick the first legal action in left-turn order."""
        import pacman
        from pacman.layout import Layout

        lay = Layout([
            "%%%%",
            "%.P%",
            "%%%%",
        ])
        state = pacman.GameState()
        state.initialize(lay, 0)

        agent = LeftTurnAgent()
        action = agent.get_action(state)
        legal = state.get_legal_pacman_actions()
        assert action in legal


# ── GreedyAgent ──────────────────────────────────────────────────────────────

class TestGreedyAgent:
    def test_init(self):
        agent = GreedyAgent()
        assert isinstance(agent, Agent)
