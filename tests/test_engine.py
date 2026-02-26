"""Tests for pacman/engine.py — game runner and GameState."""
import pytest
from pacman.engine import GameState, read_command, ClassicGameRules
from pacman.layout import Layout
from pacman.game import Directions, AgentState, Grid


# ── Helper ───────────────────────────────────────────────────────────────────

def make_game_state(layout_lines, num_ghosts=0):
    lay = Layout(layout_lines)
    state = GameState()
    state.initialize(lay, num_ghosts)
    return state


# ── GameState ────────────────────────────────────────────────────────────────

class TestGameState:
    def test_init(self):
        state = make_game_state(["%%%%", "%.P%", "%%%%"])
        assert state.get_score() == 0

    def test_get_legal_actions(self):
        state = make_game_state([
            "%%%%%",
            "%   %",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        actions = state.get_legal_actions(0)
        assert Directions.STOP in actions
        assert Directions.NORTH in actions
        assert Directions.SOUTH in actions

    def test_get_legal_pacman_actions(self):
        state = make_game_state([
            "%%%%%",
            "%   %",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        actions = state.get_legal_pacman_actions()
        assert isinstance(actions, list)
        assert len(actions) > 0

    def test_generate_pacman_successor(self):
        state = make_game_state([
            "%%%%%",
            "%   %",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        new_state = state.generate_pacman_successor(Directions.NORTH)
        assert isinstance(new_state, GameState)

    def test_get_pacman_position(self):
        state = make_game_state(["%%%%", "%.P%", "%%%%"])
        pos = state.get_pacman_position()
        assert isinstance(pos, tuple)
        assert len(pos) == 2

    def test_get_pacman_state(self):
        state = make_game_state(["%%%%", "%.P%", "%%%%"])
        pacman_state = state.get_pacman_state()
        assert isinstance(pacman_state, AgentState)
        assert pacman_state.is_pacman

    def test_get_food(self):
        state = make_game_state([
            "%%%%%",
            "%. .%",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        food = state.get_food()
        assert isinstance(food, Grid)
        assert food.count() > 0

    def test_get_walls(self):
        state = make_game_state(["%%%%", "%.P%", "%%%%"])
        walls = state.get_walls()
        assert isinstance(walls, Grid)
        assert walls[0][0] is True  # border wall

    def test_has_food(self):
        state = make_game_state([
            "%%%%%",
            "%.P %",
            "%%%%%",
        ])
        food = state.get_food()
        food_positions = food.as_list()
        if food_positions:
            x, y = food_positions[0]
            assert state.has_food(x, y)

    def test_get_num_food(self):
        state = make_game_state([
            "%%%%%",
            "%.P.%",
            "%%%%%",
        ])
        assert state.get_num_food() == 2

    def test_has_wall(self):
        state = make_game_state(["%%%%", "%.P%", "%%%%"])
        assert state.has_wall(0, 0)
        pos = state.get_pacman_position()
        assert not state.has_wall(*pos)

    def test_is_lose_is_win(self):
        state = make_game_state(["%%%%", "%.P%", "%%%%"])
        assert not state.is_lose()
        assert not state.is_win()

    def test_get_num_agents(self):
        state = make_game_state(["%%%%", "%.P%", "%%%%"])
        assert state.get_num_agents() >= 1

    def test_get_num_agents_with_ghost(self):
        state = make_game_state([
            "%%%%%%",
            "%.GP %",
            "%%%%%%",
        ], num_ghosts=1)
        assert state.get_num_agents() == 2

    def test_get_score(self):
        state = make_game_state(["%%%%", "%.P%", "%%%%"])
        assert state.get_score() == 0

    def test_deep_copy(self):
        state = make_game_state(["%%%%", "%.P%", "%%%%"])
        copy = state.deep_copy()
        assert copy.get_score() == state.get_score()
        assert copy.get_pacman_position() == state.get_pacman_position()

    def test_str(self):
        state = make_game_state(["%%%%", "%.P%", "%%%%"])
        s = str(state)
        assert "Score" in s

    def test_equality(self):
        s1 = make_game_state(["%%%%", "%.P%", "%%%%"])
        s2 = make_game_state(["%%%%", "%.P%", "%%%%"])
        assert s1 == s2

    def test_hash(self):
        s1 = make_game_state(["%%%%", "%.P%", "%%%%"])
        s2 = make_game_state(["%%%%", "%.P%", "%%%%"])
        assert hash(s1) == hash(s2)

    def test_generate_successor(self):
        state = make_game_state([
            "%%%%%",
            "%   %",
            "% P %",
            "%   %",
            "%%%%%",
        ])
        new_state = state.generate_successor(0, Directions.NORTH)
        assert isinstance(new_state, GameState)
        # Pacman should have moved north
        old_pos = state.get_pacman_position()
        new_pos = new_state.get_pacman_position()
        assert new_pos[1] == old_pos[1] + 1

    def test_illegal_action_raises(self):
        state = make_game_state(["%%%%", "%P %", "%%%%"])
        # North should be a wall
        with pytest.raises(Exception):
            state.generate_successor(0, Directions.NORTH)


# ── read_command ─────────────────────────────────────────────────────────────

class TestReadCommand:
    def test_default_args(self):
        args = read_command(['-q', '-p', 'LeftTurnAgent'])
        assert args is not None
        assert 'layout' in args

    def test_custom_layout(self):
        args = read_command(['-q', '-p', 'LeftTurnAgent', '-l', 'smallMaze'])
        assert args['layout'] is not None

    def test_num_games(self):
        args = read_command(['-q', '-p', 'LeftTurnAgent', '-n', '5'])
        assert args['num_games'] == 5

    def test_quiet_text_graphics(self):
        args = read_command(['-q', '-p', 'LeftTurnAgent', '--text_graphics'])
        assert args is not None


# ── ClassicGameRules ─────────────────────────────────────────────────────────

class TestClassicGameRules:
    def test_init(self):
        rules = ClassicGameRules(timeout=30)
        assert rules is not None

    def test_get_max_total_time(self):
        rules = ClassicGameRules(timeout=30)
        assert rules.get_max_total_time(0) > 0

    def test_get_max_startup_time(self):
        rules = ClassicGameRules(timeout=30)
        assert rules.get_max_startup_time(0) > 0

    def test_get_move_timeout(self):
        rules = ClassicGameRules(timeout=30)
        assert rules.get_move_timeout(0) > 0
