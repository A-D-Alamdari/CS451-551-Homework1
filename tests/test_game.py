"""Tests for pacman/game.py — core game framework."""
import pytest
from pacman.game import (
    Agent,
    Directions,
    Configuration,
    AgentState,
    Grid,
    Actions,
    GameStateData,
    Game,
    reconstitute_grid,
)


# ── Directions ───────────────────────────────────────────────────────────────

class TestDirections:
    def test_direction_constants(self):
        assert Directions.NORTH == "North"
        assert Directions.SOUTH == "South"
        assert Directions.EAST == "East"
        assert Directions.WEST == "West"
        assert Directions.STOP == "Stop"

    def test_left_mapping(self):
        assert Directions.LEFT[Directions.NORTH] == Directions.WEST
        assert Directions.LEFT[Directions.SOUTH] == Directions.EAST
        assert Directions.LEFT[Directions.EAST] == Directions.NORTH
        assert Directions.LEFT[Directions.WEST] == Directions.SOUTH
        assert Directions.LEFT[Directions.STOP] == Directions.STOP

    def test_right_is_inverse_of_left(self):
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            right = Directions.RIGHT[direction]
            assert Directions.LEFT[right] == direction

    def test_reverse(self):
        assert Directions.REVERSE[Directions.NORTH] == Directions.SOUTH
        assert Directions.REVERSE[Directions.SOUTH] == Directions.NORTH
        assert Directions.REVERSE[Directions.EAST] == Directions.WEST
        assert Directions.REVERSE[Directions.WEST] == Directions.EAST
        assert Directions.REVERSE[Directions.STOP] == Directions.STOP


# ── Configuration ────────────────────────────────────────────────────────────

class TestConfiguration:
    def test_get_position(self):
        c = Configuration((3, 4), Directions.NORTH)
        assert c.get_position() == (3, 4)

    def test_get_direction(self):
        c = Configuration((3, 4), Directions.EAST)
        assert c.get_direction() == Directions.EAST

    def test_is_integer(self):
        assert Configuration((1, 2), Directions.NORTH).is_integer()
        assert not Configuration((1.5, 2), Directions.NORTH).is_integer()
        assert not Configuration((1, 2.5), Directions.NORTH).is_integer()

    def test_equality(self):
        c1 = Configuration((1, 2), Directions.NORTH)
        c2 = Configuration((1, 2), Directions.NORTH)
        c3 = Configuration((1, 2), Directions.SOUTH)
        assert c1 == c2
        assert c1 != c3
        assert c1 is not None

    def test_hash(self):
        c1 = Configuration((1, 2), Directions.NORTH)
        c2 = Configuration((1, 2), Directions.NORTH)
        assert hash(c1) == hash(c2)

    def test_generate_successor(self):
        c = Configuration((3, 4), Directions.NORTH)
        c2 = c.generate_successor((1, 0))
        assert c2.get_position() == (4, 4)
        assert c2.get_direction() == Directions.EAST

    def test_generate_successor_stop_keeps_direction(self):
        c = Configuration((3, 4), Directions.NORTH)
        c2 = c.generate_successor((0, 0))
        assert c2.get_direction() == Directions.NORTH

    def test_str(self):
        c = Configuration((1, 2), Directions.NORTH)
        assert "(x,y)=" in str(c)


# ── AgentState ───────────────────────────────────────────────────────────────

class TestAgentState:
    def test_init(self):
        config = Configuration((1, 2), Directions.NORTH)
        state = AgentState(config, True)
        assert state.is_pacman
        assert state.scared_timer == 0
        assert state.get_position() == (1, 2)
        assert state.get_direction() == Directions.NORTH

    def test_copy(self):
        config = Configuration((1, 2), Directions.NORTH)
        state = AgentState(config, True)
        state.scared_timer = 5
        copy = state.copy()
        assert copy.scared_timer == 5
        assert copy.is_pacman
        assert copy.get_position() == (1, 2)

    def test_equality(self):
        config = Configuration((1, 2), Directions.NORTH)
        s1 = AgentState(config, True)
        s2 = AgentState(config, True)
        assert s1 == s2
        s2.scared_timer = 10
        assert s1 != s2
        assert s1 is not None

    def test_str_pacman(self):
        config = Configuration((1, 2), Directions.NORTH)
        state = AgentState(config, True)
        assert "Pacman" in str(state)

    def test_str_ghost(self):
        config = Configuration((1, 2), Directions.NORTH)
        state = AgentState(config, False)
        assert "Ghost" in str(state)


# ── Grid ─────────────────────────────────────────────────────────────────────

class TestGrid:
    def test_init_false(self):
        g = Grid(3, 4, False)
        assert g.width == 3
        assert g.height == 4
        for x in range(3):
            for y in range(4):
                assert g[x][y] is False

    def test_init_true(self):
        g = Grid(2, 2, True)
        for x in range(2):
            for y in range(2):
                assert g[x][y] is True

    def test_invalid_initial_value(self):
        with pytest.raises(Exception):
            Grid(2, 2, "invalid")

    def test_setitem(self):
        g = Grid(3, 3, False)
        g[1][2] = True
        assert g[1][2] is True
        assert g[0][0] is False

    def test_count(self):
        g = Grid(3, 3, False)
        g[0][0] = True
        g[1][1] = True
        g[2][2] = True
        assert g.count(True) == 3
        assert g.count(False) == 6

    def test_as_list(self):
        g = Grid(3, 3, False)
        g[0][0] = True
        g[2][1] = True
        result = g.as_list(True)
        assert (0, 0) in result
        assert (2, 1) in result
        assert len(result) == 2

    def test_copy(self):
        g = Grid(3, 3, False)
        g[1][1] = True
        g2 = g.copy()
        g2[1][1] = False
        assert g[1][1] is True  # original unchanged

    def test_deep_copy(self):
        g = Grid(3, 3, False)
        g[1][1] = True
        g2 = g.deep_copy()
        g2[1][1] = False
        assert g[1][1] is True

    def test_shallow_copy(self):
        g = Grid(3, 3, False)
        g2 = g.shallow_copy()
        assert g2.data is g.data

    def test_equality(self):
        g1 = Grid(3, 3, False)
        g2 = Grid(3, 3, False)
        assert g1 == g2
        g1[0][0] = True
        assert g1 != g2
        assert g1 is not None

    def test_hash(self):
        g1 = Grid(3, 3, False)
        g2 = Grid(3, 3, False)
        assert hash(g1) == hash(g2)

    def test_pack_and_unpack_bits(self):
        g = Grid(4, 4, False)
        g[0][0] = True
        g[1][2] = True
        g[3][3] = True
        packed = g.pack_bits()
        g2 = Grid(4, 4, bit_representation=packed[2:])
        assert g2.width == 4
        for x in range(4):
            for y in range(4):
                assert g[x][y] == g2[x][y], f"Mismatch at ({x},{y})"

    def test_str(self):
        g = Grid(3, 3, False)
        g[0][0] = True
        s = str(g)
        assert isinstance(s, str)
        assert len(s) > 0


class TestReconstituteGrid:
    def test_non_tuple_passthrough(self):
        g = Grid(2, 2, False)
        assert reconstitute_grid(g) is g

    def test_from_tuple(self):
        g = Grid(3, 3, False)
        g[1][1] = True
        packed = g.pack_bits()
        g2 = reconstitute_grid(packed)
        assert g2[1][1] is True
        assert g2.width == 3
        assert g2.height == 3


# ── Actions ──────────────────────────────────────────────────────────────────

class TestActions:
    def test_reverse_direction(self):
        assert Actions.reverse_direction(Directions.NORTH) == Directions.SOUTH
        assert Actions.reverse_direction(Directions.SOUTH) == Directions.NORTH
        assert Actions.reverse_direction(Directions.EAST) == Directions.WEST
        assert Actions.reverse_direction(Directions.WEST) == Directions.EAST
        assert Actions.reverse_direction(Directions.STOP) == Directions.STOP

    def test_vector_to_direction(self):
        assert Actions.vector_to_direction((0, 1)) == Directions.NORTH
        assert Actions.vector_to_direction((0, -1)) == Directions.SOUTH
        assert Actions.vector_to_direction((1, 0)) == Directions.EAST
        assert Actions.vector_to_direction((-1, 0)) == Directions.WEST
        assert Actions.vector_to_direction((0, 0)) == Directions.STOP

    def test_direction_to_vector(self):
        assert Actions.direction_to_vector(Directions.NORTH) == (0, 1)
        assert Actions.direction_to_vector(Directions.SOUTH) == (0, -1)
        assert Actions.direction_to_vector(Directions.EAST) == (1, 0)
        assert Actions.direction_to_vector(Directions.WEST) == (-1, 0)
        assert Actions.direction_to_vector(Directions.STOP) == (0, 0)

    def test_direction_to_vector_with_speed(self):
        dx, dy = Actions.direction_to_vector(Directions.NORTH, 2.0)
        assert dx == 0 and dy == 2.0

    def test_get_successor(self):
        assert Actions.get_successor((3, 4), Directions.NORTH) == (3, 5)
        assert Actions.get_successor((3, 4), Directions.SOUTH) == (3, 3)
        assert Actions.get_successor((3, 4), Directions.EAST) == (4, 4)
        assert Actions.get_successor((3, 4), Directions.WEST) == (2, 4)

    def test_get_possible_actions(self):
        # Build a small 5x5 grid with walls around the border
        walls = Grid(5, 5, False)
        for x in range(5):
            walls[x][0] = True
            walls[x][4] = True
        for y in range(5):
            walls[0][y] = True
            walls[4][y] = True
        config = Configuration((2, 2), Directions.STOP)
        actions = Actions.get_possible_actions(config, walls)
        # (2,2) is in the center, all 4 neighbors are open + stop
        assert Directions.NORTH in actions
        assert Directions.SOUTH in actions
        assert Directions.EAST in actions
        assert Directions.WEST in actions

    def test_get_legal_neighbors(self):
        walls = Grid(5, 5, False)
        for x in range(5):
            walls[x][0] = True
            walls[x][4] = True
        for y in range(5):
            walls[0][y] = True
            walls[4][y] = True
        neighbors = Actions.get_legal_neighbors((2, 2), walls)
        assert (2, 3) in neighbors  # north
        assert (2, 1) in neighbors  # south
        assert (3, 2) in neighbors  # east
        assert (1, 2) in neighbors  # west


# ── GameStateData ────────────────────────────────────────────────────────────

class TestGameStateData:
    def test_init_empty(self):
        data = GameStateData()
        assert data._lose is False
        assert data._win is False
        assert data.score_change == 0

    def test_init_from_prev(self):
        from pacman.layout import Layout
        lay = Layout(["%%%%", "%.P%", "%%%%"])
        data = GameStateData()
        data.initialize(lay, 0)
        data2 = GameStateData(data)
        assert data2.score == 0
        assert len(data2.agent_states) > 0

    def test_deep_copy(self):
        from pacman.layout import Layout
        lay = Layout(["%%%%", "%.P%", "%%%%"])
        data = GameStateData()
        data.initialize(lay, 0)
        copy = data.deep_copy()
        copy.score = 100
        assert data.score == 0

    def test_equality(self):
        from pacman.layout import Layout
        lay = Layout(["%%%%", "%.P%", "%%%%"])
        d1 = GameStateData()
        d1.initialize(lay, 0)
        d2 = GameStateData()
        d2.initialize(lay, 0)
        assert d1 == d2
        assert d1 is not None

    def test_str(self):
        from pacman.layout import Layout
        lay = Layout(["%%%%", "%.P%", "%%%%"])
        data = GameStateData()
        data.initialize(lay, 0)
        s = str(data)
        assert "Score" in s

    def test_initialize(self):
        from pacman.layout import Layout
        lay = Layout(["%%%%", "%GP%", "%%%%"])
        data = GameStateData()
        data.initialize(lay, 1)
        # Should have pacman and ghost
        assert len(data.agent_states) == 2
        assert data.agent_states[0].is_pacman


# ── Agent ────────────────────────────────────────────────────────────────────

class TestAgent:
    def test_default_index(self):
        a = Agent()
        assert a.index == 0

    def test_custom_index(self):
        a = Agent(3)
        assert a.index == 3


# ── Game ─────────────────────────────────────────────────────────────────────

class TestGame:
    def test_init(self):
        class MockDisplay:
            pass

        class MockRules:
            pass

        game = Game([None], MockDisplay(), MockRules())
        assert game.game_over is False
        assert game.agent_crashed is False
        assert len(game.move_history) == 0

    def test_get_progress_not_over(self):
        class MockDisplay:
            pass

        class MockRules:
            def get_progress(self, game):
                return 0.5

        game = Game([None], MockDisplay(), MockRules())
        assert game.get_progress() == 0.5

    def test_get_progress_game_over(self):
        class MockDisplay:
            pass

        class MockRules:
            pass

        game = Game([None], MockDisplay(), MockRules())
        game.game_over = True
        assert game.get_progress() == 1.0
