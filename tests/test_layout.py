"""Tests for pacman/layout.py — maze loading and parsing."""

from pacman.layout import Layout, get_layout, try_to_load


class TestLayout:
    """Tests for the Layout class."""

    def test_basic_layout(self):
        text = [
            "%%%%",
            "%.P%",
            "%%%%",
        ]
        lay = Layout(text)
        assert lay.width == 4
        assert lay.height == 3

    def test_walls(self):
        text = [
            "%%%%",
            "%.P%",
            "%%%%",
        ]
        lay = Layout(text)
        # borders are walls
        assert lay.walls[0][0] is True
        assert lay.walls[3][0] is True
        assert lay.walls[0][2] is True
        # interior should not be walls
        assert lay.walls[1][1] is False
        assert lay.walls[2][1] is False

    def test_food(self):
        text = [
            "%%%%",
            "%.P%",
            "%%%%",
        ]
        lay = Layout(text)
        # (1,1) should have food (the '.' is at column 1, row 1 from bottom)
        assert lay.food[1][1] is True
        assert lay.total_food == 1

    def test_agent_positions(self):
        text = [
            "%%%%",
            "%GP%",
            "%%%%",
        ]
        lay = Layout(text)
        # Pacman should be first (index 0)
        assert lay.agent_positions[0][0] is True  # is_pacman=True
        assert lay.num_ghosts == 1

    def test_capsules(self):
        text = [
            "%%%%",
            "%oP%",
            "%%%%",
        ]
        lay = Layout(text)
        assert len(lay.capsules) == 1

    def test_is_wall(self):
        text = [
            "%%%%",
            "%.P%",
            "%%%%",
        ]
        lay = Layout(text)
        assert lay.is_wall((0, 0)) is True
        assert lay.is_wall((1, 1)) is False

    def test_deep_copy(self):
        text = [
            "%%%%",
            "%.P%",
            "%%%%",
        ]
        lay = Layout(text)
        copy = lay.deep_copy()
        assert copy.width == lay.width
        assert copy.height == lay.height
        # Modify copy and check original is unaffected
        copy.walls[1][1] = True
        assert lay.walls[1][1] is False

    def test_str(self):
        text = [
            "%%%%",
            "%.P%",
            "%%%%",
        ]
        lay = Layout(text)
        s = str(lay)
        assert "%" in s

    def test_get_num_ghosts(self):
        text = [
            "%%%%%",
            "%G.G%",
            "%.P.%",
            "%%%%%",
        ]
        lay = Layout(text)
        assert lay.get_num_ghosts() == 2

    def test_get_random_legal_position(self):
        text = [
            "%%%%",
            "%.P%",
            "%%%%",
        ]
        lay = Layout(text)
        # There are only 2 non-wall positions: (1,1) and (2,1)
        pos = lay.get_random_legal_position()
        assert not lay.is_wall(pos)

    def test_get_random_corner(self):
        text = [
            "%%%%%%",
            "%    %",
            "%  P %",
            "%    %",
            "%%%%%%",
        ]
        lay = Layout(text)
        corner = lay.get_random_corner()
        expected_corners = [(1, 1), (1, 3), (4, 1), (4, 3)]
        assert corner in expected_corners

    def test_get_furthest_corner(self):
        text = [
            "%%%%%%",
            "%    %",
            "%  P %",
            "%    %",
            "%%%%%%",
        ]
        lay = Layout(text)
        # Pacman is at (3, 2), furthest corner should be (1, 1) or (1,3)
        corner = lay.get_furthest_corner((3, 2))
        assert corner in [(1, 1), (1, 3), (4, 1), (4, 3)]

    def test_numbered_ghosts(self):
        text = [
            "%%%%%%",
            "%1 2P%",
            "%%%%%%",
        ]
        lay = Layout(text)
        assert lay.num_ghosts == 2

    def test_layout_text_preserved(self):
        text = [
            "%%%%",
            "%.P%",
            "%%%%",
        ]
        lay = Layout(text)
        assert lay.layout_text == text


class TestGetLayout:
    def test_nonexistent_layout(self):
        result = get_layout("definitely_does_not_exist_xyz", back=0)
        assert result is None


class TestTryToLoad:
    def test_nonexistent_file(self):
        result = try_to_load("nonexistent_layout.lay")
        assert result is None
