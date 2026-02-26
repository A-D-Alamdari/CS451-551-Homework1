"""Tests for pacman/eight_puzzle.py — the 8-puzzle domain."""
import pytest
from pacman.eight_puzzle import (
    EightPuzzleState,
    EightPuzzleSearchProblem,
    EIGHT_PUZZLE_DATA,
    load_eight_puzzle,
    create_random_eight_puzzle,
)


# ── EightPuzzleState ─────────────────────────────────────────────────────────

class TestEightPuzzleState:
    def test_goal_state(self):
        state = EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8])
        assert state.is_goal()

    def test_non_goal_state(self):
        state = EightPuzzleState([1, 0, 2, 3, 4, 5, 6, 7, 8])
        assert not state.is_goal()

    def test_blank_location(self):
        state = EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8])
        assert state.blank_location == (0, 0)

    def test_blank_location_center(self):
        state = EightPuzzleState([1, 2, 3, 4, 0, 5, 6, 7, 8])
        assert state.blank_location == (1, 1)

    def test_legal_moves_top_left(self):
        state = EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8])
        moves = state.legal_moves()
        assert "down" in moves
        assert "right" in moves
        assert "up" not in moves
        assert "left" not in moves

    def test_legal_moves_center(self):
        state = EightPuzzleState([1, 2, 3, 4, 0, 5, 6, 7, 8])
        moves = state.legal_moves()
        assert len(moves) == 4
        assert set(moves) == {"up", "down", "left", "right"}

    def test_legal_moves_bottom_right(self):
        state = EightPuzzleState([1, 2, 3, 4, 5, 6, 7, 8, 0])
        moves = state.legal_moves()
        assert "up" in moves
        assert "left" in moves
        assert "down" not in moves
        assert "right" not in moves

    def test_result_right(self):
        state = EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8])
        new_state = state.result("right")
        assert new_state.blank_location == (0, 1)
        assert new_state.cells[0][0] == 1  # swapped with blank
        assert new_state.cells[0][1] == 0

    def test_result_down(self):
        state = EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8])
        new_state = state.result("down")
        assert new_state.blank_location == (1, 0)
        assert new_state.cells[1][0] == 0

    def test_result_does_not_modify_original(self):
        state = EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8])
        original_cells = [row[:] for row in state.cells]
        state.result("right")
        assert state.cells == original_cells

    def test_result_illegal_move(self):
        state = EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8])
        with pytest.raises(ValueError):
            state.result("invalid")

    def test_equality(self):
        s1 = EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8])
        s2 = EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8])
        assert s1 == s2

    def test_inequality(self):
        s1 = EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8])
        s2 = EightPuzzleState([1, 0, 2, 3, 4, 5, 6, 7, 8])
        assert s1 != s2

    def test_hash(self):
        s1 = EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8])
        s2 = EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8])
        assert hash(s1) == hash(s2)

    def test_str(self):
        state = EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8])
        s = str(state)
        assert "---" in s
        assert "|" in s

    def test_result_left_from_goal(self):
        # Goal has blank at (0,0), left is illegal
        state = EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8])
        assert "left" not in state.legal_moves()

    def test_round_trip(self):
        """Moving right then left should return to original state."""
        state = EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8])
        new_state = state.result("right").result("left")
        assert state == new_state


# ── EightPuzzleSearchProblem ─────────────────────────────────────────────────

class TestEightPuzzleSearchProblem:
    def test_is_goal_state_on_goal(self):
        goal = EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8])
        prob = EightPuzzleSearchProblem(goal)
        assert prob.is_goal_state(goal)

    def test_is_goal_state_on_non_goal(self):
        non_goal = EightPuzzleState([1, 0, 2, 3, 4, 5, 6, 7, 8])
        prob = EightPuzzleSearchProblem(non_goal)
        assert not prob.is_goal_state(non_goal)

    def test_get_successors(self):
        state = EightPuzzleState([1, 2, 3, 4, 0, 5, 6, 7, 8])
        prob = EightPuzzleSearchProblem(state)
        succs = prob.get_successors(state)
        assert len(succs) == 4  # center blank has 4 moves
        for new_state, action, cost in succs:
            assert isinstance(new_state, EightPuzzleState)
            assert action in ["up", "down", "left", "right"]
            assert cost == 1

    def test_get_cost_of_actions(self):
        state = EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8])
        prob = EightPuzzleSearchProblem(state)
        assert prob.get_cost_of_actions(["right", "down"]) == 2
        assert prob.get_cost_of_actions([]) == 0


# ── Loading puzzles ──────────────────────────────────────────────────────────

class TestLoadEightPuzzle:
    def test_load_puzzle_0(self):
        puzzle = load_eight_puzzle(0)
        assert isinstance(puzzle, EightPuzzleState)
        assert puzzle.cells[0] == [1, 0, 2]

    def test_load_all_puzzles(self):
        for i in range(len(EIGHT_PUZZLE_DATA)):
            puzzle = load_eight_puzzle(i)
            assert isinstance(puzzle, EightPuzzleState)

    def test_load_out_of_range(self):
        with pytest.raises(IndexError):
            load_eight_puzzle(100)


class TestCreateRandomEightPuzzle:
    def test_create_random_puzzle(self):
        puzzle = create_random_eight_puzzle(10)
        assert isinstance(puzzle, EightPuzzleState)
        # Should contain all numbers 0-8
        all_nums = set()
        for row in puzzle.cells:
            for num in row:
                all_nums.add(num)
        assert all_nums == set(range(9))

    def test_zero_moves_is_goal(self):
        puzzle = create_random_eight_puzzle(0)
        assert puzzle.is_goal()
