"""
Pac-Man Search Assignment — Student Self-Check Tests
=====================================================

Run these tests with pytest to verify your implementations:

    pytest tests/check.py -v                  # Run all checks
    pytest tests/check.py -v -k "q1"          # Run only Q1 (DFS)
    pytest tests/check.py -v -k "q2"          # Run only Q2 (BFS)
    pytest tests/check.py -v -k "q3"          # Run only Q3 (UCS)
    pytest tests/check.py -v -k "q4"          # Run only Q4 (A*)
    pytest tests/check.py -v -k "q5"          # Run only Q5 (Corners Problem)
    pytest tests/check.py -v -k "q6"          # Run only Q6 (Corners Heuristic)
    pytest tests/check.py -v -k "q7"          # Run only Q7 (Food Heuristic)
    pytest tests/check.py -v -k "q8"          # Run only Q8 (Closest Dot)

Each test class corresponds to one assignment question (Q1–Q8, 75 points total).
Tests are ordered so that foundational algorithms are checked first.
"""

import pytest
import sys
import os

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pacman
from pacman.layout import Layout
from pacman.game import Directions, Actions
from pacman.search import (
    SearchProblem,
    depth_first_search,
    breadth_first_search,
    uniform_cost_search,
    a_star_search,
    null_heuristic,
)
from pacman.search_agents import (
    PositionSearchProblem,
    CornersProblem,
    corners_heuristic,
    FoodSearchProblem,
    food_heuristic,
    AnyFoodSearchProblem,
    ClosestDotSearchAgent,
    manhattan_heuristic,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_game_state(layout_lines):
    """Create a GameState from a list of layout strings."""
    lay = Layout(layout_lines)
    state = pacman.GameState()
    state.initialize(lay, 0)
    return state


class GraphProblem(SearchProblem):
    """A configurable graph-based search problem for testing."""

    def __init__(self, graph_text):
        self.expanded_states = []
        lines = graph_text.strip().split("\n")
        self.start = lines[0].split(":")[1].strip()
        self.goals = lines[1].split(":")[1].strip().split()
        self.successors = {}
        for line in lines[2:]:
            parts = line.split()
            if len(parts) == 3:
                src, action, dst = parts
                cost = 1.0
            elif len(parts) == 4:
                src, action, dst, cost = parts
                cost = float(cost)
            else:
                continue
            self.successors.setdefault(src, []).append((dst, action, cost))

    def get_start_state(self):
        return self.start

    def is_goal_state(self, state):
        return state in self.goals

    def get_successors(self, state):
        self.expanded_states.append(state)
        return list(self.successors.get(state, []))

    def get_cost_of_actions(self, actions):
        cost = 0
        state = self.start
        for a in actions:
            for next_state, action, step_cost in self.successors.get(state, []):
                if action == a:
                    cost += step_cost
                    state = next_state
                    break
        return cost


def get_states_from_path(start, path):
    """Trace a path through a Pac-Man layout, returning visited positions."""
    visited = [start]
    curr = start
    for a in path:
        x, y = curr
        dx, dy = Actions.direction_to_vector(a)
        curr = (int(x + dx), int(y + dy))
        visited.append(curr)
    return visited


def validate_path(problem, path):
    """Check that a path is a valid solution for a search problem."""
    state = problem.get_start_state()
    for action in path:
        found = False
        for succ, act, cost in problem.get_successors(state):
            if act == action:
                state = succ
                found = True
                break
        if not found:
            return False
    return problem.is_goal_state(state)


# ═══════════════════════════════════════════════════════════════════════════════
# Q1 — Depth-First Search (9 points)
# ═══════════════════════════════════════════════════════════════════════════════

class TestQ1_DepthFirstSearch:
    """Q1: Depth-First Search — search the deepest nodes first (9 pts)."""

    def test_q1_returns_list(self):
        """DFS must return a list of actions."""
        problem = GraphProblem(
            "start_state: A\ngoal_states: B\nA go_B B 1.0"
        )
        result = depth_first_search(problem)
        assert isinstance(result, list), (
            "DFS must return a list of actions, got %s" % type(result)
        )

    def test_q1_simple_path(self):
        """DFS finds a path in a simple linear graph: A -> B -> C."""
        problem = GraphProblem(
            "start_state: A\ngoal_states: C\nA go_B B 1.0\nB go_C C 1.0"
        )
        result = depth_first_search(problem)
        assert result == ["go_B", "go_C"], (
            "Expected ['go_B', 'go_C'], got %s" % result
        )

    def test_q1_graph_backtrack(self):
        """DFS correctly backtracks — returns correct action sequence."""
        problem = GraphProblem(
            "start_state: A\ngoal_states: G\n"
            "A 0:A->B B 1.0\nA 1:A->C C 2.0\nA 2:A->D D 4.0\nC 0:C->G G 8.0"
        )
        result = depth_first_search(problem)
        assert validate_path(problem, result), (
            "DFS returned an invalid path: %s" % result
        )

    def test_q1_handles_cycles(self):
        """DFS uses graph search (visited set) — doesn't loop on cycles."""
        problem = GraphProblem(
            "start_state: A\ngoal_states: C\n"
            "A go_B B 1.0\nB go_A A 1.0\nB go_C C 1.0"
        )
        result = depth_first_search(problem)
        assert validate_path(problem, result), (
            "DFS must handle cycles via graph search. Got: %s" % result
        )

    def test_q1_start_is_goal(self):
        """DFS returns empty path when start state is the goal."""
        problem = GraphProblem("start_state: G\ngoal_states: G")
        result = depth_first_search(problem)
        assert result == [], (
            "When start is goal, DFS should return [], got %s" % result
        )

    def test_q1_pacman_tiny_maze(self):
        """DFS solves tinyMaze — path leads Pac-Man to the food."""
        state = make_game_state([
            "%%%%%%%",
            "%    P%",
            "% %%% %",
            "%  %  %",
            "%%   %%",
            "%. %%%%",
            "%%%%%%%",
        ])
        problem = PositionSearchProblem(state, warn=False)
        path = depth_first_search(problem)
        assert isinstance(path, list) and len(path) > 0, "DFS must find a path"
        # Verify the path actually reaches the goal
        pos = state.get_pacman_position()
        walls = state.get_walls()
        for action in path:
            dx, dy = Actions.direction_to_vector(action)
            pos = (int(pos[0] + dx), int(pos[1] + dy))
            assert not walls[pos[0]][pos[1]], "Path goes through a wall!"
        assert problem.is_goal_state(pos), "Path does not reach the goal"


# ═══════════════════════════════════════════════════════════════════════════════
# Q2 — Breadth-First Search (9 points)
# ═══════════════════════════════════════════════════════════════════════════════

class TestQ2_BreadthFirstSearch:
    """Q2: Breadth-First Search — find the shortest path (9 pts)."""

    def test_q2_returns_list(self):
        """BFS must return a list of actions."""
        problem = GraphProblem(
            "start_state: A\ngoal_states: B\nA go_B B 1.0"
        )
        result = breadth_first_search(problem)
        assert isinstance(result, list), (
            "BFS must return a list, got %s" % type(result)
        )

    def test_q2_finds_shortest_path(self):
        """BFS finds the shortest (fewest actions) path."""
        # A has two paths to G: A->B->G (2 steps) and A->C->D->G (3 steps)
        problem = GraphProblem(
            "start_state: A\ngoal_states: G\n"
            "A go_B B 1.0\nA go_C C 1.0\nB go_G G 1.0\n"
            "C go_D D 1.0\nD go_G G 1.0"
        )
        result = breadth_first_search(problem)
        assert len(result) == 2, (
            "BFS should find the 2-step path, got %d steps: %s"
            % (len(result), result)
        )

    def test_q2_handles_cycles(self):
        """BFS uses graph search — doesn't loop on cycles."""
        problem = GraphProblem(
            "start_state: A\ngoal_states: C\n"
            "A go_B B 1.0\nB go_A A 1.0\nB go_C C 1.0"
        )
        result = breadth_first_search(problem)
        assert validate_path(problem, result), (
            "BFS must handle cycles. Got: %s" % result
        )

    def test_q2_start_is_goal(self):
        """BFS returns empty path when start is already the goal."""
        problem = GraphProblem("start_state: G\ngoal_states: G")
        result = breadth_first_search(problem)
        assert result == [], (
            "When start is goal, BFS should return [], got %s" % result
        )

    def test_q2_bfs_vs_dfs(self):
        """BFS guarantees shortest path even when DFS would find a longer one."""
        problem = GraphProblem(
            "start_state: A\ngoal_states: G\n"
            "A 0:A->B B 1.0\nB 0:B->C C 1.0\nC 0:C->G G 1.0\nA 1:A->G G 1.0"
        )
        result = breadth_first_search(problem)
        assert len(result) == 1, (
            "BFS should find the 1-step path A->G, got %d steps: %s"
            % (len(result), result)
        )

    def test_q2_pacman_tiny_maze(self):
        """BFS finds optimal path on tinyMaze."""
        state = make_game_state([
            "%%%%%%%",
            "%    P%",
            "% %%% %",
            "%  %  %",
            "%%   %%",
            "%. %%%%",
            "%%%%%%%",
        ])
        problem = PositionSearchProblem(state, warn=False)
        path = breadth_first_search(problem)
        # BFS optimal path for tinyMaze is known to be length 8
        assert len(path) == 8, (
            "BFS optimal path on tinyMaze should be 8 steps, got %d" % len(path)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Q3 — Uniform-Cost Search (9 points)
# ═══════════════════════════════════════════════════════════════════════════════

class TestQ3_UniformCostSearch:
    """Q3: Uniform-Cost Search — find the least-cost path (9 pts)."""

    def test_q3_returns_list(self):
        """UCS must return a list of actions."""
        problem = GraphProblem(
            "start_state: A\ngoal_states: B\nA go_B B 1.0"
        )
        result = uniform_cost_search(problem)
        assert isinstance(result, list), (
            "UCS must return a list, got %s" % type(result)
        )

    def test_q3_cheapest_not_shortest(self):
        """UCS picks the cheapest path, even if it has more steps."""
        # A->D costs 10 (1 step), A->B->D costs 2 (2 steps)
        problem = GraphProblem(
            "start_state: A\ngoal_states: D\n"
            "A go_B B 1.0\nA go_D D 10.0\nB go_D D 1.0"
        )
        result = uniform_cost_search(problem)
        cost = problem.get_cost_of_actions(result)
        assert cost == 2, (
            "UCS should find cost-2 path (A->B->D), got cost %s: %s"
            % (cost, result)
        )

    def test_q3_weighted_graph(self):
        """UCS finds optimal path in a graph with varying edge costs."""
        problem = GraphProblem(
            "start_state: A\ngoal_states: H F\n"
            "A Right B 2.0\nB Right H 4.0\nB Down D 1.0\n"
            "B Up C 2.0\nB Left A 2.0\nC Down B 2.0\n"
            "D Right E 2.5\nD Down F 2.0\nD Left G 1.5"
        )
        result = uniform_cost_search(problem)
        assert result == ["Right", "Down", "Down"], (
            "UCS optimal path should be ['Right', 'Down', 'Down'], got %s"
            % result
        )

    def test_q3_handles_cycles(self):
        """UCS handles cycles correctly via graph search."""
        problem = GraphProblem(
            "start_state: A\ngoal_states: C\n"
            "A go_B B 1.0\nB go_A A 1.0\nB go_C C 1.0"
        )
        result = uniform_cost_search(problem)
        assert validate_path(problem, result)

    def test_q3_start_is_goal(self):
        """UCS returns empty path when start is the goal."""
        problem = GraphProblem("start_state: G\ngoal_states: G")
        result = uniform_cost_search(problem)
        assert result == []

    def test_q3_pacman_tiny_maze(self):
        """UCS solves tinyMaze optimally (all costs = 1, so same as BFS)."""
        state = make_game_state([
            "%%%%%%%",
            "%    P%",
            "% %%% %",
            "%  %  %",
            "%%   %%",
            "%. %%%%",
            "%%%%%%%",
        ])
        problem = PositionSearchProblem(state, warn=False)
        path = uniform_cost_search(problem)
        assert len(path) == 8, (
            "UCS on tinyMaze (unit costs) should match BFS: 8 steps, got %d"
            % len(path)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Q4 — A* Search (9 points)
# ═══════════════════════════════════════════════════════════════════════════════

class TestQ4_AStarSearch:
    """Q4: A* Search — guided search with heuristic (9 pts)."""

    def test_q4_returns_list(self):
        """A* must return a list of actions."""
        problem = GraphProblem(
            "start_state: A\ngoal_states: B\nA go_B B 1.0"
        )
        result = a_star_search(problem)
        assert isinstance(result, list), (
            "A* must return a list, got %s" % type(result)
        )

    def test_q4_with_null_heuristic(self):
        """A* with null heuristic behaves like UCS."""
        problem = GraphProblem(
            "start_state: A\ngoal_states: D\n"
            "A go_B B 1.0\nA go_D D 10.0\nB go_D D 1.0"
        )
        result = a_star_search(problem, heuristic=null_heuristic)
        cost = problem.get_cost_of_actions(result)
        assert cost == 2, (
            "A* with null heuristic should find cost-2 path, got cost %s" % cost
        )

    def test_q4_with_heuristic(self):
        """A* uses heuristic to guide search efficiently."""
        problem = GraphProblem(
            "start_state: S\ngoal_states: G\n"
            "S 0 A 2.0\nS 1 B 3.0\nS 2 D 5.0\n"
            "A 0 C 3.0\nA 1 S 2.0\nB 0 D 4.0\nB 1 S 3.0\n"
            "C 0 A 3.0\nC 1 D 1.0\nC 2 G 2.0\n"
            "D 0 B 4.0\nD 1 C 1.0\nD 2 G 5.0\nD 3 S 5.0"
        )
        heuristic_values = {"S": 6.0, "A": 2.5, "B": 5.25, "C": 1.125, "D": 1.0625, "G": 0}

        def heuristic(state, problem=None):
            return heuristic_values.get(state, 0)

        result = a_star_search(problem, heuristic=heuristic)
        assert result == ["0", "0", "2"], (
            "A* with given heuristic should find path ['0', '0', '2'] "
            "(S->A->C->G), got %s" % result
        )

    def test_q4_handles_cycles(self):
        """A* handles cycles correctly."""
        problem = GraphProblem(
            "start_state: A\ngoal_states: C\n"
            "A go_B B 1.0\nB go_A A 1.0\nB go_C C 1.0"
        )
        result = a_star_search(problem)
        assert validate_path(problem, result)

    def test_q4_start_is_goal(self):
        """A* returns empty path when start is the goal."""
        problem = GraphProblem("start_state: G\ngoal_states: G")
        result = a_star_search(problem)
        assert result == []

    def test_q4_pacman_with_manhattan(self):
        """A* with Manhattan heuristic solves tinyMaze optimally."""
        state = make_game_state([
            "%%%%%%%",
            "%    P%",
            "% %%% %",
            "%  %  %",
            "%%   %%",
            "%. %%%%",
            "%%%%%%%",
        ])
        problem = PositionSearchProblem(state, warn=False)
        path = a_star_search(problem, heuristic=manhattan_heuristic)
        assert len(path) == 8, (
            "A* with Manhattan heuristic on tinyMaze: expected 8 steps, got %d"
            % len(path)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Q5 — Corners Problem (9 points)
# ═══════════════════════════════════════════════════════════════════════════════

class TestQ5_CornersProblem:
    """Q5: Corners Problem — state representation with BFS (9 pts).

    Requires: Q2 (BFS).
    """

    def _make_corners_problem(self, layout_lines):
        state = make_game_state(layout_lines)
        return CornersProblem(state), state

    def test_q5_get_start_state(self):
        """CornersProblem.get_start_state() returns a hashable state."""
        problem, _ = self._make_corners_problem([
            "%%%%%%",
            "%.  .%",
            "%  P %",
            "%.  .%",
            "%%%%%%",
        ])
        start = problem.get_start_state()
        assert start is not None, "get_start_state() should not return None"
        # State must be hashable (for use in visited sets)
        try:
            hash(start)
        except TypeError:
            pytest.fail("Start state must be hashable (use tuples, not lists)")

    def test_q5_start_not_goal(self):
        """Start state is not the goal (not all corners visited yet)."""
        problem, _ = self._make_corners_problem([
            "%%%%%%",
            "%.  .%",
            "%  P %",
            "%.  .%",
            "%%%%%%",
        ])
        start = problem.get_start_state()
        assert not problem.is_goal_state(start), (
            "Start state should not be a goal — Pac-Man hasn't visited all corners"
        )

    def test_q5_get_successors(self):
        """get_successors() returns valid (state, action, cost) triples."""
        problem, _ = self._make_corners_problem([
            "%%%%%%",
            "%.  .%",
            "%  P %",
            "%.  .%",
            "%%%%%%",
        ])
        start = problem.get_start_state()
        successors = problem.get_successors(start)
        assert len(successors) > 0, "Start should have at least one successor"
        for succ_state, action, cost in successors:
            assert action in [Directions.NORTH, Directions.SOUTH,
                              Directions.EAST, Directions.WEST], (
                "Action must be a Direction, got %s" % action
            )
            assert cost == 1, "Step cost should be 1, got %s" % cost
            try:
                hash(succ_state)
            except TypeError:
                pytest.fail("Successor states must be hashable")

    def test_q5_tiny_corners(self):
        """BFS solves tinyCorners — visits all 4 corners with optimal length."""
        problem, game_state = self._make_corners_problem([
            "%%%%%%%%",
            "%.    .%",
            "%   P  %",
            "% %%%% %",
            "% %    %",
            "% % %%%%",
            "%.%   .%",
            "%%%%%%%%",
        ])
        path = breadth_first_search(problem)
        assert isinstance(path, list), "BFS must return a list"
        assert len(path) == 28, (
            "Optimal path for tinyCorners is 28 steps, got %d" % len(path)
        )
        # Verify all corners are visited
        visited = get_states_from_path(game_state.get_pacman_position(), path)
        walls = game_state.get_walls()
        top, right = walls.height - 2, walls.width - 2
        corners = [(1, 1), (1, top), (right, 1), (right, top)]
        for corner in corners:
            assert corner in visited, (
                "Path must visit corner %s but didn't" % (corner,)
            )

    def test_q5_successor_updates_corners(self):
        """Moving to a corner should update the corner-visited info in state."""
        problem, game_state = self._make_corners_problem([
            "%%%%%%",
            "%.  .%",
            "%  P %",
            "%.  .%",
            "%%%%%%",
        ])
        # Use BFS to find a solution, then verify goal is reached
        path = breadth_first_search(problem)
        assert isinstance(path, list) and len(path) > 0, (
            "BFS should find a path to visit all corners"
        )
        # After following the path, the final state should be a goal
        state = problem.get_start_state()
        for action in path:
            for succ, act, cost in problem.get_successors(state):
                if act == action:
                    state = succ
                    break
        assert problem.is_goal_state(state), (
            "After following BFS path, final state should be a goal"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Q6 — Corners Heuristic (9 points)
# ═══════════════════════════════════════════════════════════════════════════════

class TestQ6_CornersHeuristic:
    """Q6: Corners Heuristic — admissible & consistent heuristic (9 pts).

    Requires: Q4 (A*), Q5 (CornersProblem).
    Grading based on node expansion:
        ≤ 2000 nodes → 3 pts, ≤ 1600 → 6 pts, ≤ 1200 → 9 pts.
    """

    def _setup(self, layout_lines):
        state = make_game_state(layout_lines)
        problem = CornersProblem(state)
        return problem

    def test_q6_heuristic_returns_number(self):
        """corners_heuristic returns a non-negative number."""
        problem = self._setup([
            "%%%%%%",
            "%.  .%",
            "%P   %",
            "%.  .%",
            "%%%%%%",
        ])
        start = problem.get_start_state()
        h = corners_heuristic(start, problem)
        assert isinstance(h, (int, float)), (
            "Heuristic must return a number, got %s" % type(h)
        )
        assert h >= 0, "Heuristic must be non-negative, got %s" % h

    def test_q6_heuristic_nonzero_at_start(self):
        """Heuristic should be non-trivial (> 0) when corners remain."""
        problem = self._setup([
            "%%%%%%",
            "%.  .%",
            "%P   %",
            "%.  .%",
            "%%%%%%",
        ])
        start = problem.get_start_state()
        h = corners_heuristic(start, problem)
        assert h > 0, (
            "Heuristic should be > 0 at start (corners unvisited), got %s" % h
        )

    def test_q6_admissibility(self):
        """Heuristic must be admissible: h(start) ≤ true optimal cost."""
        problem = self._setup([
            "%%%%%%",
            "%.  .%",
            "%P   %",
            "%.  .%",
            "%%%%%%",
        ])
        start = problem.get_start_state()
        h = corners_heuristic(start, problem)
        # Find actual optimal cost with BFS
        optimal_path = breadth_first_search(problem)
        true_cost = len(optimal_path)
        assert h <= true_cost, (
            "Heuristic is inadmissible! h=%s > true_cost=%s" % (h, true_cost)
        )

    def test_q6_consistency(self):
        """Heuristic must be consistent: h(n) - h(n') ≤ step_cost."""
        problem = self._setup([
            "%%%%%%",
            "%.  .%",
            "%P   %",
            "%.  .%",
            "%%%%%%",
        ])
        start = problem.get_start_state()
        h0 = corners_heuristic(start, problem)
        for succ, action, cost in problem.get_successors(start):
            h1 = corners_heuristic(succ, problem)
            assert h0 - h1 <= cost, (
                "Inconsistent heuristic! h(n)=%s, h(n')=%s, step_cost=%s. "
                "h(n)-h(n')=%s > step_cost" % (h0, h1, cost, h0 - h1)
            )

    def test_q6_goal_heuristic_zero(self):
        """Heuristic at goal state must be 0."""
        problem = self._setup([
            "%%%%%%",
            "%.  .%",
            "%P   %",
            "%.  .%",
            "%%%%%%",
        ])
        # Find a goal state by running A*
        path = a_star_search(problem, corners_heuristic)
        state = problem.get_start_state()
        for action in path:
            for succ, act, cost in problem.get_successors(state):
                if act == action:
                    state = succ
                    break
        assert problem.is_goal_state(state), "A* should reach a goal state"
        h = corners_heuristic(state, problem)
        assert h == 0, "Heuristic at goal must be 0, got %s" % h

    def test_q6_medium_corners(self):
        """A* with corners heuristic solves mediumCorners — check quality.

        Full marks thresholds: ≤2000 (3 pts), ≤1600 (6 pts), ≤1200 (9 pts).
        """
        problem = self._setup([
            "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",
            "%.      % % %              %.%",
            "%       % % %%%%%% %%%%%%% % %",
            "%       %        %     % %   %",
            "%%%%% %%%%% %%% %% %%%%% % %%%",
            "%   % % % %   %    %     %   %",
            "% %%% % % % %%%%%%%% %%% %%% %",
            "%       %     %%     % % %   %",
            "%%% % %%%%%%% %%%% %%% % % % %",
            "% %           %%     %     % %",
            "% % %%%%% % %%%% % %%% %%% % %",
            "%   %     %      % %   % %%% %",
            "%.  %P%%%%%      % %%% %    .%",
            "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",
        ])
        path = a_star_search(problem, corners_heuristic)
        assert validate_path(problem, path), "A* returned an invalid path"
        expanded = problem._expanded
        cost = len(path)
        assert cost == 106, (
            "Optimal cost for mediumCorners is 106, got %d" % cost
        )
        # Report grade based on expansion thresholds
        if expanded <= 1200:
            grade = "9/9"
        elif expanded <= 1600:
            grade = "6/9"
        elif expanded <= 2000:
            grade = "3/9"
        else:
            grade = "0/9 (heuristic is too weak)"
        print(
            "\n  [Q6] mediumCorners: expanded %d nodes → %s"
            % (expanded, grade)
        )
        assert expanded <= 2000, (
            "Too many nodes expanded: %d (need ≤ 2000 for any credit)" % expanded
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Q7 — Food Heuristic (12 points)
# ═══════════════════════════════════════════════════════════════════════════════

class TestQ7_FoodHeuristic:
    """Q7: Food Search Heuristic — admissible heuristic for eating all food (12 pts).

    Requires: Q4 (A*).
    Grading based on node expansion on trickySearch:
        any solution → 3 pts, ≤15000 → 6 pts, ≤12000 → 9 pts, ≤7000 → 12 pts.
    """

    def _setup(self, layout_lines):
        state = make_game_state(layout_lines)
        problem = FoodSearchProblem(state)
        return problem

    def test_q7_heuristic_returns_number(self):
        """food_heuristic returns a non-negative number."""
        problem = self._setup([
            "%%%%%",
            "%. .%",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        start = problem.get_start_state()
        h = food_heuristic(start, problem)
        assert isinstance(h, (int, float)), (
            "Heuristic must return a number, got %s" % type(h)
        )
        assert h >= 0, "Heuristic must be non-negative, got %s" % h

    def test_q7_goal_heuristic_zero(self):
        """Heuristic at goal (no food left) must be 0."""
        problem = self._setup([
            "%%%%%",
            "%   %",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        start = problem.get_start_state()
        assert problem.is_goal_state(start), "No food → should be goal"
        h = food_heuristic(start, problem)
        assert h == 0, "Heuristic at goal must be 0, got %s" % h

    def test_q7_admissibility(self):
        """Heuristic is admissible: h(start) ≤ true optimal cost."""
        problem = self._setup([
            "%%%%%%",
            "%....%",
            "%....%",
            "%P...%",
            "%%%%%%",
        ])
        start = problem.get_start_state()
        h = food_heuristic(start, problem)
        # Find actual cost with UCS
        path = uniform_cost_search(problem)
        true_cost = problem.get_cost_of_actions(path)
        assert h <= true_cost, (
            "Heuristic is inadmissible! h=%s > true_cost=%s" % (h, true_cost)
        )

    def test_q7_consistency(self):
        """Heuristic is consistent: h(n) - h(n') ≤ step_cost."""
        problem = self._setup([
            "%%%%%%",
            "%....%",
            "%....%",
            "%P...%",
            "%%%%%%",
        ])
        start = problem.get_start_state()
        h0 = food_heuristic(start, problem)
        for succ, action, cost in problem.get_successors(start):
            h1 = food_heuristic(succ, problem)
            assert h0 - h1 <= cost, (
                "Inconsistent heuristic! h(n)=%s, h(n')=%s, step_cost=%s"
                % (h0, h1, cost)
            )

    def test_q7_tricky_search(self):
        """A* with food heuristic solves trickySearch — check quality.

        Full marks thresholds:
            any solution → 3/12, ≤15000 → 6/12, ≤12000 → 9/12, ≤7000 → 12/12.
        """
        problem = self._setup([
            "%%%%%%%%%%%%%%%%%%%%",
            "%.           ..%   %",
            "%.%%.%%.%%.%%.%% % %",
            "%        P       % %",
            "%%%%%%%%%%%%%%%%%% %",
            "%.....             %",
            "%%%%%%%%%%%%%%%%%%%%",
        ])
        path = a_star_search(problem, food_heuristic)
        assert validate_path(problem, path), "A* returned an invalid path"
        expanded = problem._expanded
        if expanded <= 7000:
            grade = "12/12"
        elif expanded <= 12000:
            grade = "9/12"
        elif expanded <= 15000:
            grade = "6/12"
        else:
            grade = "3/12 (solution found, heuristic needs improvement)"
        print(
            "\n  [Q7] trickySearch: expanded %d nodes → %s"
            % (expanded, grade)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Q8 — Closest Dot Search (9 points)
# ═══════════════════════════════════════════════════════════════════════════════

class TestQ8_ClosestDotSearch:
    """Q8: Find path to the closest food dot (9 pts).

    Tests AnyFoodSearchProblem.is_goal_state and
    ClosestDotSearchAgent.find_path_to_closest_dot.
    """

    def test_q8_any_food_goal_at_food(self):
        """AnyFoodSearchProblem: position with food IS a goal."""
        state = make_game_state([
            "%%%%%",
            "%. .%",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        problem = AnyFoodSearchProblem(state)
        food_positions = state.get_food().as_list()
        assert len(food_positions) > 0, "Layout should have food"
        for pos in food_positions:
            assert problem.is_goal_state(pos), (
                "Position %s has food but is_goal_state returned False" % (pos,)
            )

    def test_q8_any_food_no_food_not_goal(self):
        """AnyFoodSearchProblem: position without food is NOT a goal."""
        state = make_game_state([
            "%%%%%",
            "%. .%",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        problem = AnyFoodSearchProblem(state)
        food = state.get_food()
        # Pac-Man's position should not have food
        pac_pos = state.get_pacman_position()
        if not food[pac_pos[0]][pac_pos[1]]:
            assert not problem.is_goal_state(pac_pos), (
                "Position %s has no food but is_goal_state returned True"
                % (pac_pos,)
            )

    def test_q8_find_closest_dot_simple(self):
        """ClosestDotSearchAgent finds path to the nearest food."""
        state = make_game_state([
            "%%%%%%",
            "%....%",
            "%....%",
            "%P...%",
            "%%%%%%",
        ])
        agent = ClosestDotSearchAgent()
        path = agent.find_path_to_closest_dot(state)
        assert isinstance(path, list), (
            "find_path_to_closest_dot must return a list, got %s" % type(path)
        )
        # The closest dot is 1 step away (East or North)
        assert len(path) == 1, (
            "Closest dot is 1 step away, got path of length %d: %s"
            % (len(path), path)
        )

    def test_q8_find_closest_dot_farther(self):
        """ClosestDotSearchAgent finds nearest dot even when farther away."""
        state = make_game_state([
            "%%%%%%",
            "%   .%",
            "%    %",
            "%P   %",
            "%%%%%%",
        ])
        agent = ClosestDotSearchAgent()
        path = agent.find_path_to_closest_dot(state)
        assert isinstance(path, list), "Must return a list"
        # Food is at (4,3), Pacman at (1,1); BFS shortest is 5 steps
        assert len(path) == 5, (
            "Closest dot should be 5 steps away, got %d" % len(path)
        )

    def test_q8_full_agent(self):
        """ClosestDotSearchAgent collects all food via repeated closest-dot search."""
        state = make_game_state([
            "%%%%%%",
            "%....%",
            "%....%",
            "%P...%",
            "%%%%%%",
        ])
        agent = ClosestDotSearchAgent()
        agent.register_initial_state(state)
        assert len(agent.actions) > 0, "Agent should find actions to collect food"
