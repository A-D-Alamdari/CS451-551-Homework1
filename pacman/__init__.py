# pacman/__init__.py
# Re-export key symbols from engine so that `import pacman; pacman.GameState`
# and `from pacman import GameState` keep working after the restructure.

from pacman.engine import GameState, run_games, read_command, load_agent, replay_game
from pacman.game import Directions, Agent, Actions, Grid
