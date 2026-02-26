"""Tests for pacman/search_agents.py — search problem definitions and agents."""
import pytest
import pacman
from pacman.layout import Layout
from pacman.game import Directions
from pacman.search_agents import (
    GoWestAgent,
    SearchAgent,
    PositionSearchProblem,
    StayEastSearchAgent,
    StayWestSearchAgent,
    manhattan_heuristic,
    euclidean_heuristic,
    CornersProblem,
    corners_heuristic,
    FoodSearchProblem,
    food_heuristic,
    AnyFoodSearchProblem,
    AStarCornersAgent,
    AStarFoodSearchAgent,
)


# ── Helper to create a GameState from layout text ────────────────────────────

def make_game_state(layout_lines):
    """Create a GameState from a list of layout lines."""
    lay = Layout(layout_lines)
    state = pacman.GameState()
    state.initialize(lay, 0)
    return state


# ── GoWestAgent ──────────────────────────────────────────────────────────────

class TestGoWestAgent:
    def test_goes_west(self):
        state = make_game_state([
            "%%%%",
            "%.P%",
            "%%%%",
        ])
        agent = GoWestAgent()
        action = agent.get_action(state)
        assert action == Directions.WEST

    def test_stops_at_wall(self):
        state = make_game_state([
            "%%%",
            "%P%",
            "%%%",
        ])
        agent = GoWestAgent()
        action = agent.get_action(state)
        # Can't go west (wall), should stop
        assert action == Directions.STOP


# ── PositionSearchProblem ────────────────────────────────────────────────────

class TestPositionSearchProblem:
    def test_get_start_state(self):
        state = make_game_state([
            "%%%%%",
            "%  .%",
            "% %%%",
            "%P  %",
            "%%%%%",
        ])
        prob = PositionSearchProblem(state, warn=False)
        start = prob.get_start_state()
        assert isinstance(start, tuple)
        assert len(start) == 2

    def test_is_goal_state(self):
        state = make_game_state([
            "%%%%%",
            "%  .%",
            "% %%%",
            "%P  %",
            "%%%%%",
        ])
        prob = PositionSearchProblem(state, goal=(1, 1), warn=False)
        assert prob.is_goal_state((1, 1))
        assert not prob.is_goal_state((2, 2))

    def test_get_successors(self):
        state = make_game_state([
            "%%%%%",
            "%   %",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        prob = PositionSearchProblem(state, warn=False)
        start = prob.get_start_state()
        successors = prob.get_successors(start)
        assert len(successors) > 0
        for next_state, action, cost in successors:
            assert isinstance(next_state, tuple)
            assert action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
            assert cost >= 0

    def test_expanded_counter(self):
        state = make_game_state([
            "%%%%%",
            "%   %",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        prob = PositionSearchProblem(state, warn=False)
        assert prob._expanded == 0
        prob.get_successors(prob.get_start_state())
        assert prob._expanded == 1

    def test_get_cost_of_actions_none(self):
        state = make_game_state([
            "%%%%%",
            "%   %",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        prob = PositionSearchProblem(state, warn=False)
        assert prob.get_cost_of_actions(None) == 999999

    def test_get_cost_of_actions_valid(self):
        state = make_game_state([
            "%%%%%",
            "%   %",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        prob = PositionSearchProblem(state, warn=False)
        cost = prob.get_cost_of_actions([Directions.NORTH])
        assert cost == 1

    def test_custom_cost_function(self):
        state = make_game_state([
            "%%%%%",
            "%   %",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        prob = PositionSearchProblem(state, cost_fn=lambda pos: 5, warn=False)
        cost = prob.get_cost_of_actions([Directions.NORTH])
        assert cost == 5


# ── Heuristics ───────────────────────────────────────────────────────────────

class TestHeuristics:
    def test_manhattan_heuristic(self):
        class FakeProblem:
            goal = (5, 5)

        assert manhattan_heuristic((0, 0), FakeProblem()) == 10
        assert manhattan_heuristic((5, 5), FakeProblem()) == 0
        assert manhattan_heuristic((3, 4), FakeProblem()) == 3

    def test_euclidean_heuristic(self):
        class FakeProblem:
            goal = (0, 0)

        assert euclidean_heuristic((0, 0), FakeProblem()) == 0.0
        assert abs(euclidean_heuristic((3, 4), FakeProblem()) - 5.0) < 1e-9

    def test_manhattan_admissibility(self):
        """Manhattan distance should always be <= true cost (admissible)."""

        class FakeProblem:
            goal = (5, 5)

        # The true shortest path on a grid (no walls) is the Manhattan distance
        h = manhattan_heuristic((0, 0), FakeProblem())
        true_cost = 10  # no walls: walk 5 right + 5 up
        assert h <= true_cost

    def test_euclidean_admissibility(self):
        """Euclidean distance <= Manhattan distance for any point."""

        class FakeProblem:
            goal = (5, 5)

        e = euclidean_heuristic((0, 0), FakeProblem())
        m = manhattan_heuristic((0, 0), FakeProblem())
        assert e <= m


# ── CornersProblem ───────────────────────────────────────────────────────────

class TestCornersProblem:
    def test_init(self):
        state = make_game_state([
            "%%%%%%",
            "%.  .%",
            "%  P %",
            "%.  .%",
            "%%%%%%",
        ])
        prob = CornersProblem(state)
        assert prob._expanded == 0
        assert len(prob.corners) == 4

    def test_get_start_state_raises(self):
        """Student stub calls raise_not_defined."""
        state = make_game_state([
            "%%%%%%",
            "%.  .%",
            "%  P %",
            "%.  .%",
            "%%%%%%",
        ])
        prob = CornersProblem(state)
        with pytest.raises(SystemExit):
            prob.get_start_state()

    def test_get_cost_of_actions(self):
        state = make_game_state([
            "%%%%%%",
            "%.  .%",
            "%  P %",
            "%.  .%",
            "%%%%%%",
        ])
        prob = CornersProblem(state)
        assert prob.get_cost_of_actions(None) == 999999
        cost = prob.get_cost_of_actions([Directions.NORTH])
        assert cost == 1

    def test_corners_heuristic_trivial(self):
        """Default corners_heuristic returns 0."""
        state = make_game_state([
            "%%%%%%",
            "%.  .%",
            "%  P %",
            "%.  .%",
            "%%%%%%",
        ])
        prob = CornersProblem(state)
        # We can't call get_start_state (not implemented), so test with a mock state
        h = corners_heuristic("any_state", prob)
        assert h == 0


# ── FoodSearchProblem ────────────────────────────────────────────────────────

class TestFoodSearchProblem:
    def test_get_start_state(self):
        state = make_game_state([
            "%%%%%",
            "%. .%",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        prob = FoodSearchProblem(state)
        start = prob.get_start_state()
        pos, food_grid = start
        assert isinstance(pos, tuple)
        assert food_grid.count() > 0

    def test_is_goal_state(self):
        state = make_game_state([
            "%%%%%",
            "%. .%",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        prob = FoodSearchProblem(state)
        start = prob.get_start_state()
        assert not prob.is_goal_state(start)

    def test_is_goal_state_no_food(self):
        state = make_game_state([
            "%%%%%",
            "%   %",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        prob = FoodSearchProblem(state)
        start = prob.get_start_state()
        assert prob.is_goal_state(start)

    def test_get_successors(self):
        state = make_game_state([
            "%%%%%",
            "%. .%",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        prob = FoodSearchProblem(state)
        start = prob.get_start_state()
        succs = prob.get_successors(start)
        assert len(succs) > 0
        for (pos, food), action, cost in succs:
            assert cost == 1

    def test_expanded_counter(self):
        state = make_game_state([
            "%%%%%",
            "%. .%",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        prob = FoodSearchProblem(state)
        assert prob._expanded == 0
        prob.get_successors(prob.get_start_state())
        assert prob._expanded == 1

    def test_heuristic_info(self):
        state = make_game_state([
            "%%%%%",
            "%. .%",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        prob = FoodSearchProblem(state)
        assert isinstance(prob.heuristic_info, dict)

    def test_food_heuristic_trivial(self):
        """Default food_heuristic returns 0."""
        state = make_game_state([
            "%%%%%",
            "%. .%",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        prob = FoodSearchProblem(state)
        start = prob.get_start_state()
        assert food_heuristic(start, prob) == 0


# ── AnyFoodSearchProblem ─────────────────────────────────────────────────────

class TestAnyFoodSearchProblem:
    def test_init(self):
        state = make_game_state([
            "%%%%%",
            "%. .%",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        prob = AnyFoodSearchProblem(state)
        assert prob.start_state == state.get_pacman_position()

    def test_is_goal_state_raises(self):
        """Student stub calls raise_not_defined."""
        state = make_game_state([
            "%%%%%",
            "%. .%",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        prob = AnyFoodSearchProblem(state)
        with pytest.raises(SystemExit):
            prob.is_goal_state(prob.start_state)


# ── SearchAgent ──────────────────────────────────────────────────────────────

class TestSearchAgent:
    def test_invalid_function(self):
        with pytest.raises(AttributeError, match="is not a search function"):
            SearchAgent(fn="nonexistent_function")

    def test_invalid_problem(self):
        with pytest.raises(AttributeError, match="is not a search problem type"):
            SearchAgent(fn="depth_first_search", prob="NonexistentProblem")


# ── StayEastSearchAgent / StayWestSearchAgent ────────────────────────────────

class TestStayAgents:
    def test_stay_east_init(self):
        agent = StayEastSearchAgent()
        assert agent.search_function is not None
        assert agent.search_type is not None

    def test_stay_west_init(self):
        agent = StayWestSearchAgent()
        assert agent.search_function is not None
        assert agent.search_type is not None


# ── AStarCornersAgent / AStarFoodSearchAgent ─────────────────────────────────

class TestAStarAgents:
    def test_astar_corners_init(self):
        agent = AStarCornersAgent()
        assert agent.search_function is not None
        assert agent.search_type is CornersProblem

    def test_astar_food_init(self):
        agent = AStarFoodSearchAgent()
        assert agent.search_function is not None
        assert agent.search_type is FoodSearchProblem
