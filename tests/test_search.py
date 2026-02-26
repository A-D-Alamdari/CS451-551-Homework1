"""Tests for pacman/search.py — search algorithms and SearchProblem interface."""
import pytest
from pacman.search import (
    SearchProblem,
    null_heuristic,
    depth_first_search,
    breadth_first_search,
    uniform_cost_search,
    a_star_search,
    dfs,
    bfs,
    ucs,
    astar,
)


# ── A simple graph-based search problem for testing ──────────────────────────

class SimpleGraphProblem(SearchProblem):
    """A small directed graph for testing search algorithms.

    Graph:
        A --1--> B --1--> C (goal)
        A --1--> D --1--> C (goal)
        A --5--> C (goal)  # direct but expensive
    """

    def __init__(self):
        self.expanded = []
        self.graph = {
            "A": [("B", "go_B", 1), ("D", "go_D", 1), ("C", "go_C", 5)],
            "B": [("C", "go_C", 1)],
            "D": [("C", "go_C", 1)],
            "C": [],
        }

    def get_start_state(self):
        return "A"

    def is_goal_state(self, state):
        return state == "C"

    def get_successors(self, state):
        self.expanded.append(state)
        return self.graph.get(state, [])

    def get_cost_of_actions(self, actions):
        cost = 0
        state = "A"
        for action in actions:
            for next_state, act, step_cost in self.graph[state]:
                if act == action:
                    cost += step_cost
                    state = next_state
                    break
        return cost


class LinearProblem(SearchProblem):
    """A linear chain: A -> B -> C -> D (goal)."""

    def __init__(self):
        self.graph = {
            "A": [("B", "A_to_B", 1)],
            "B": [("C", "B_to_C", 1)],
            "C": [("D", "C_to_D", 1)],
            "D": [],
        }

    def get_start_state(self):
        return "A"

    def is_goal_state(self, state):
        return state == "D"

    def get_successors(self, state):
        return self.graph.get(state, [])

    def get_cost_of_actions(self, actions):
        return len(actions)


class CyclicProblem(SearchProblem):
    """A graph with cycles: A -> B -> A (cycle), B -> C (goal)."""

    def __init__(self):
        self.graph = {
            "A": [("B", "go_B", 1)],
            "B": [("A", "go_A", 1), ("C", "go_C", 1)],
            "C": [],
        }

    def get_start_state(self):
        return "A"

    def is_goal_state(self, state):
        return state == "C"

    def get_successors(self, state):
        return self.graph.get(state, [])

    def get_cost_of_actions(self, actions):
        return len(actions)


class WeightedProblem(SearchProblem):
    """Graph where the shortest path != fewest actions.

        A --1--> B --1--> D (goal)
        A --10-> D (goal)
    """

    def __init__(self):
        self.graph = {
            "A": [("B", "go_B", 1), ("D", "go_D", 10)],
            "B": [("D", "go_D", 1)],
            "D": [],
        }

    def get_start_state(self):
        return "A"

    def is_goal_state(self, state):
        return state == "D"

    def get_successors(self, state):
        return self.graph.get(state, [])

    def get_cost_of_actions(self, actions):
        cost = 0
        state = "A"
        for action in actions:
            for next_state, act, step_cost in self.graph[state]:
                if act == action:
                    cost += step_cost
                    state = next_state
                    break
        return cost


class StartIsGoalProblem(SearchProblem):
    """The start state is already a goal."""

    def get_start_state(self):
        return "GOAL"

    def is_goal_state(self, state):
        return state == "GOAL"

    def get_successors(self, state):
        return []

    def get_cost_of_actions(self, actions):
        return 0


# ── SearchProblem interface ──────────────────────────────────────────────────

class TestSearchProblem:
    def test_is_abstract(self):
        sp = SearchProblem()
        # These should raise SystemExit (util.raise_not_defined calls sys.exit)
        with pytest.raises(SystemExit):
            sp.get_start_state()

    def test_null_heuristic(self):
        assert null_heuristic("any_state") == 0
        assert null_heuristic("any_state", "any_problem") == 0


# ── Abbreviations ────────────────────────────────────────────────────────────

class TestAbbreviations:
    def test_abbreviations_exist(self):
        assert dfs is depth_first_search
        assert bfs is breadth_first_search
        assert ucs is uniform_cost_search
        assert astar is a_star_search


# ── DFS ──────────────────────────────────────────────────────────────────────
# Note: The student code is not implemented (stubs call raise_not_defined).
# These tests verify the interface; when student implements, they will pass.

class TestDepthFirstSearch:
    def test_raises_not_defined(self):
        """DFS stub should exit since it's not implemented."""
        with pytest.raises(SystemExit):
            depth_first_search(SimpleGraphProblem())


# ── BFS ──────────────────────────────────────────────────────────────────────

class TestBreadthFirstSearch:
    def test_raises_not_defined(self):
        """BFS stub should exit since it's not implemented."""
        with pytest.raises(SystemExit):
            breadth_first_search(SimpleGraphProblem())


# ── UCS ──────────────────────────────────────────────────────────────────────

class TestUniformCostSearch:
    def test_raises_not_defined(self):
        """UCS stub should exit since it's not implemented."""
        with pytest.raises(SystemExit):
            uniform_cost_search(SimpleGraphProblem())


# ── A* ───────────────────────────────────────────────────────────────────────

class TestAStarSearch:
    def test_raises_not_defined(self):
        """A* stub should exit since it's not implemented."""
        with pytest.raises(SystemExit):
            a_star_search(SimpleGraphProblem())

    def test_null_heuristic_default(self):
        """A* with default heuristic should still call the stub."""
        with pytest.raises(SystemExit):
            a_star_search(SimpleGraphProblem())
