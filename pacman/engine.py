"""
engine.py
---------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).

Modified and extended by Amin D. Alamdari (amin.alamdari@ozu.edu.tr), 2026.
Changes: Project restructuring, modernized Python packaging, and updated
assignment scaffolding. See README.md for full list of changes.
"""
from __future__ import annotations

"""
Pacman.py holds the logic for the classic pacman game along with the main
code to run a game.  This file is divided into three sections:

  (i)  Your interface to the pacman world:
          Pacman is a complex environment.  You probably don't want to
          read through all the code we wrote to make the game runs
          correctly.  This section contains the parts of the code
          that you will need to understand in order to complete the
          project.  There is also some code in game.py that you should
          understand.

  (ii)  The hidden secrets of pacman:
          This section contains all the logic code that the pacman
          environment uses to decide who can move where, who dies when
          things collide, etc.  You shouldn't need to read this section
          of code, but you can if you want.

  (iii) Framework to start a game:
          The final section contains the code for reading the command
          you use to set up the game, then starting up a new game, along with
          linking in all the external parts (agent functions, graphics).
          Check this section out to see all the options available to you.

To play your first game, type 'python main.py' from the command line.
The keys are 'a', 's', 'd', and 'w' to move (or arrow keys).  Have fun!
"""
from typing import Any, Optional
from pacman.game import GameStateData
from pacman.game import Game
from pacman.game import Directions
from pacman.game import Actions
from pacman.game import Agent, AgentState, Grid
from pacman.util import nearest_point
from pacman.util import manhattan_distance
from pacman.layout import Layout
import pacman.layout as layout_module
import sys
import random


###################################################
# YOUR INTERFACE TO THE PACMAN WORLD: A GameState #
###################################################

class GameState:
    """
    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes.

    GameStates are used by the Game object to capture the actual state of the game and
    can be used by agents to reason about the game.

    Much of the information in a GameState is stored in a GameStateData object.  We
    strongly suggest that you access that data via the accessor methods below rather
    than referring to the GameStateData object directly.

    Note that in classic Pacman, Pacman is always agent 0.
    """

    ####################################################
    # Accessor methods: use these to access state data #
    ####################################################

    # static variable keeps track of which states have had get_legal_actions called
    explored: set[GameState] = set()

    def get_and_reset_explored() -> set[GameState]:
        tmp = GameState.explored.copy()
        GameState.explored = set()
        return tmp

    get_and_reset_explored = staticmethod(get_and_reset_explored)

    def get_legal_actions(self, agent_index: int = 0) -> list[str]:
        """
        Returns the legal actions for the agent specified.
        """
        #        GameState.explored.add(self)
        if self.is_win() or self.is_lose(): return []

        if agent_index == 0:  # Pacman is moving
            return PacmanRules.get_legal_actions(self)
        else:
            return GhostRules.get_legal_actions(self, agent_index)

    def generate_successor(self, agent_index: int, action: str) -> GameState:
        """
        Returns the successor state after the specified agent takes the action.
        """
        # Check that successors exist
        if self.is_win() or self.is_lose(): raise Exception('Can\'t generate a successor of a terminal state.')

        # Copy current state
        state = GameState(self)

        # Let agent's logic deal with its action's effects on the board
        if agent_index == 0:  # Pacman is moving
            state.data._eaten = [False for _i in range(state.get_num_agents())]
            PacmanRules.apply_action(state, action)
        else:  # A ghost is moving
            GhostRules.apply_action(state, action, agent_index)

        # Time passes
        if agent_index == 0:
            state.data.score_change += -TIME_PENALTY  # Penalty for waiting around
        else:
            GhostRules.decrement_timer(state.data.agent_states[agent_index])

        # Resolve multi-agent effects
        GhostRules.check_death(state, agent_index)

        # Bookkeeping
        state.data._agent_moved = agent_index
        state.data.score += state.data.score_change
        GameState.explored.add(self)
        GameState.explored.add(state)
        return state

    def get_legal_pacman_actions(self) -> list[str]:
        return self.get_legal_actions(0)

    def generate_pacman_successor(self, action: str) -> GameState:
        """
        Generates the successor state after the specified pacman move
        """
        return self.generate_successor(0, action)

    def get_pacman_state(self) -> AgentState:
        """
        Returns an AgentState object for pacman (in game.py)

        state.pos gives the current position
        state.direction gives the travel vector
        """
        return self.data.agent_states[0].copy()

    def get_pacman_position(self) -> Optional[tuple[float, float]]:
        return self.data.agent_states[0].get_position()

    def get_ghost_states(self) -> list[AgentState]:
        return self.data.agent_states[1:]

    def get_ghost_state(self, agent_index: int) -> AgentState:
        if agent_index == 0 or agent_index >= self.get_num_agents():
            raise Exception("Invalid index passed to get_ghost_state")
        return self.data.agent_states[agent_index]

    def get_ghost_position(self, agent_index: int) -> Optional[tuple[float, float]]:
        if agent_index == 0:
            raise Exception("Pacman's index passed to get_ghost_position")
        return self.data.agent_states[agent_index].get_position()

    def get_ghost_positions(self) -> list[Optional[tuple[float, float]]]:
        return [s.get_position() for s in self.get_ghost_states()]

    def get_num_agents(self) -> int:
        return len(self.data.agent_states)

    def get_score(self) -> float:
        return float(self.data.score)

    def get_capsules(self) -> list[tuple[int, int]]:
        """
        Returns a list of positions (x,y) of the remaining capsules.
        """
        return self.data.capsules

    def get_num_food(self) -> int:
        return self.data.food.count()

    def get_food(self) -> Grid:
        """
        Returns a Grid of boolean food indicator variables.

        Grids can be accessed via list notation, so to check
        if there is food at (x,y), just call

        current_food = state.get_food()
        if current_food[x][y] == True: ...
        """
        return self.data.food

    def get_walls(self) -> Grid:
        """
        Returns a Grid of boolean wall indicator variables.

        Grids can be accessed via list notation, so to check
        if there is a wall at (x,y), just call

        walls = state.get_walls()
        if walls[x][y] == True: ...
        """
        return self.data.layout.walls

    def has_food(self, x: int, y: int) -> bool:
        return self.data.food[x][y]

    def has_wall(self, x: int, y: int) -> bool:
        return self.data.layout.walls[x][y]

    def is_lose(self) -> bool:
        return getattr(self.data, '_lose', False)

    def is_win(self) -> bool:
        return getattr(self.data, '_win', False)

    #############################################
    #             Helper methods:               #
    # You shouldn't need to call these directly #
    #############################################

    def __init__(self, prev_state: Optional[GameState] = None) -> None:
        """
        Generates a new state by copying information from its predecessor.
        """
        if prev_state is not None:  # Initial state
            self.data = GameStateData(prev_state.data)
        else:
            self.data = GameStateData()

    def deep_copy(self) -> GameState:
        state = GameState(self)
        state.data = self.data.deep_copy()
        return state

    def __eq__(self, other: object) -> bool:
        """
        Allows two states to be compared.
        """
        return hasattr(other, 'data') and self.data == other.data

    def __hash__(self) -> int:
        """
        Allows states to be keys of dictionaries.
        """
        return hash(self.data)

    def __str__(self) -> str:
        return str(self.data)

    def initialize(self, game_layout: Layout, num_ghost_agents: int = 1000) -> None:
        """
        Creates an initial game state from a layout array (see layout.py).
        """
        self.data.initialize(game_layout, num_ghost_agents)


############################################################################
#                     THE HIDDEN SECRETS OF PACMAN                         #
#                                                                          #
# You shouldn't need to look through the code in this section of the file. #
############################################################################

SCARED_TIME: int = 40  # Moves ghosts are scared
COLLISION_TOLERANCE: float = 0.7  # How close ghosts must be to Pacman to kill
TIME_PENALTY: int = 1  # Number of points lost each round


class ClassicGameRules:
    """
    These game rules manage the control flow of a game, deciding when
    and how the game starts and ends.
    """

    def __init__(self, timeout: int = 30) -> None:
        self.timeout: int = timeout
        self.initial_state: Optional[GameState] = None
        self.quiet: bool = False

    def new_game(self, game_layout: Layout, pacman_agent: Agent, ghost_agents: list[Agent], display: Any,
                 quiet: bool = False, catch_exceptions: bool = False) -> Game:
        agents = [pacman_agent] + ghost_agents[:game_layout.get_num_ghosts()]
        init_state = GameState()
        init_state.initialize(game_layout, len(ghost_agents))
        game = Game(agents, display, self, catch_exceptions=catch_exceptions)
        game.state = init_state
        self.initial_state = init_state.deep_copy()
        self.quiet = quiet
        return game

    def process(self, state: GameState, game: Game) -> None:
        """
        Checks to see whether it is time to end the game.
        """
        if state.is_win(): self.win(state, game)
        if state.is_lose(): self.lose(state, game)

    def win(self, state: GameState, game: Game) -> None:
        if not self.quiet: print("Pacman emerges victorious! Score: %d" % state.data.score)
        game.game_over = True

    def lose(self, state: GameState, game: Game) -> None:
        if not self.quiet: print("Pacman died! Score: %d" % state.data.score)
        game.game_over = True

    def get_progress(self, game: Game) -> float:
        return float(game.state.get_num_food()) / self.initial_state.get_num_food()

    @staticmethod
    def agent_crash(_game: Game, agent_index: int) -> None:
        if agent_index == 0:
            print("Pacman crashed")
        else:
            print("A ghost crashed")

    def get_max_total_time(self, _agent_index: int) -> int:
        return self.timeout

    def get_max_startup_time(self, _agent_index: int) -> int:
        return self.timeout

    def get_move_warning_time(self, _agent_index: int) -> int:
        return self.timeout

    def get_move_timeout(self, _agent_index: int) -> int:
        return self.timeout

    @staticmethod
    def get_max_time_warnings(_agent_index: int) -> int:
        return 0


class PacmanRules:
    """
    These functions govern how pacman interacts with his environment under
    the classic game rules.
    """
    PACMAN_SPEED: int = 1

    def get_legal_actions(state: GameState) -> list[str]:
        """
        Returns a list of possible actions.
        """
        return Actions.get_possible_actions(state.get_pacman_state().configuration, state.data.layout.walls)

    get_legal_actions = staticmethod(get_legal_actions)

    def apply_action(state: GameState, action: str) -> None:
        """
        Edits the state to reflect the results of the action.
        """
        legal = PacmanRules.get_legal_actions(state)
        if action not in legal:
            raise Exception("Illegal action " + str(action))

        pacman_state = state.data.agent_states[0]

        # Update Configuration
        vector = Actions.direction_to_vector(action, PacmanRules.PACMAN_SPEED)
        pacman_state.configuration = pacman_state.configuration.generate_successor(vector)

        # Eat
        next_pos = pacman_state.configuration.get_position()
        nearest = nearest_point(next_pos)
        if manhattan_distance(nearest, next_pos) <= 0.5:
            # Remove food
            PacmanRules.consume(nearest, state)

    apply_action = staticmethod(apply_action)

    def consume(position: tuple[int, int], state: GameState) -> None:
        x, y = position
        # Eat food
        if state.data.food[x][y]:
            state.data.score_change += 10
            state.data.food = state.data.food.copy()
            state.data.food[x][y] = False
            state.data._food_eaten = position
            # TODO: cache num_food?
            num_food = state.get_num_food()
            if num_food == 0 and not getattr(state.data, '_lose', False):
                state.data.score_change += 500
                state.data._win = True
        # Eat capsule
        if position in state.get_capsules():
            state.data.capsules.remove(position)
            state.data._capsule_eaten = position
            # Reset all ghosts' scared timers
            for index in range(1, len(state.data.agent_states)):
                state.data.agent_states[index].scared_timer = SCARED_TIME

    consume = staticmethod(consume)


class GhostRules:
    """
    These functions dictate how ghosts interact with their environment.
    """
    GHOST_SPEED: float = 1.0

    def get_legal_actions(state: GameState, ghost_index: int) -> list[str]:
        """
        Ghosts cannot stop, and cannot turn around unless they
        reach a dead end, but can turn 90 degrees at intersections.
        """
        conf = state.get_ghost_state(ghost_index).configuration
        possible_actions = Actions.get_possible_actions(conf, state.data.layout.walls)
        reverse = Actions.reverse_direction(conf.direction)
        if Directions.STOP in possible_actions:
            possible_actions.remove(Directions.STOP)
        if reverse in possible_actions and len(possible_actions) > 1:
            possible_actions.remove(reverse)
        return possible_actions

    get_legal_actions = staticmethod(get_legal_actions)

    def apply_action(state: GameState, action: str, ghost_index: int) -> None:

        legal = GhostRules.get_legal_actions(state, ghost_index)
        if action not in legal:
            raise Exception("Illegal ghost action " + str(action))

        ghost_state = state.data.agent_states[ghost_index]
        speed = GhostRules.GHOST_SPEED
        if ghost_state.scared_timer > 0: speed /= 2.0
        vector = Actions.direction_to_vector(action, speed)
        ghost_state.configuration = ghost_state.configuration.generate_successor(vector)

    apply_action = staticmethod(apply_action)

    def decrement_timer(ghost_state: AgentState) -> None:
        timer = ghost_state.scared_timer
        if timer == 1:
            ghost_state.configuration.pos = nearest_point(ghost_state.configuration.pos)
        ghost_state.scared_timer = max(0, timer - 1)

    decrement_timer = staticmethod(decrement_timer)

    def check_death(state: GameState, agent_index: int) -> None:
        pacman_position = state.get_pacman_position()
        if agent_index == 0:  # Pacman just moved; Anyone can kill him
            for index in range(1, len(state.data.agent_states)):
                ghost_state = state.data.agent_states[index]
                ghost_position = ghost_state.configuration.get_position()
                if GhostRules.can_kill(pacman_position, ghost_position):
                    GhostRules.collide(state, ghost_state, index)
        else:
            ghost_state = state.data.agent_states[agent_index]
            ghost_position = ghost_state.configuration.get_position()
            if GhostRules.can_kill(pacman_position, ghost_position):
                GhostRules.collide(state, ghost_state, agent_index)

    check_death = staticmethod(check_death)

    def collide(state: GameState, ghost_state: AgentState, agent_index: int) -> None:
        if ghost_state.scared_timer > 0:
            state.data.score_change += 200
            GhostRules.place_ghost(state, ghost_state)
            ghost_state.scared_timer = 0
            # Added for first-person
            state.data._eaten[agent_index] = True
        else:
            if not getattr(state.data, '_win', False):
                state.data.score_change -= 500
                state.data._lose = True

    collide = staticmethod(collide)

    def can_kill(pacman_position: tuple[float, float], ghost_position: tuple[float, float]) -> bool:
        return manhattan_distance(ghost_position, pacman_position) <= COLLISION_TOLERANCE

    can_kill = staticmethod(can_kill)

    def place_ghost(state: GameState, ghost_state: AgentState) -> None:
        ghost_state.configuration = ghost_state.start

    place_ghost = staticmethod(place_ghost)


#############################
# FRAMEWORK TO START A GAME #
#############################

def default(s: str) -> str:
    return s + ' [Default: %default]'


def parse_agent_args(s: Optional[str]) -> dict[str, Any]:
    if s is None: return {}
    pieces = s.split(',')
    opts: dict[str, Any] = {}
    for p in pieces:
        if '=' in p:
            key, val = p.split('=')
        else:
            key, val = p, 1
        opts[key] = val
    return opts


def read_command(argv: list[str]) -> dict[str, Any]:
    """
    Processes the command used to run pacman from the command line.
    """
    from optparse import OptionParser
    usage_str = """
    USAGE:      python main.py <options>
    EXAMPLES:   (1) python main.py
                    - starts an interactive game
                (2) python main.py --layout smallClassic --zoom 2
                OR  python main.py -l smallClassic -z 2
                    - starts an interactive game on a smaller board, zoomed in
    """
    parser = OptionParser(usage_str)

    parser.add_option('-n', '--num_games', dest='num_games', type='int',
                      help=default('the number of GAMES to play'), metavar='GAMES', default=1)
    parser.add_option('-l', '--layout', dest='layout',
                      help=default('the LAYOUT_FILE from which to load the map layout'),
                      metavar='LAYOUT_FILE', default='mediumClassic')
    parser.add_option('-p', '--pacman', dest='pacman',
                      help=default('the agent TYPE in the pacman_agents module to use'),
                      metavar='TYPE', default='KeyboardAgent')
    parser.add_option('-t', '--text_graphics', action='store_true', dest='text_graphics',
                      help='Display output as text only', default=False)
    parser.add_option('-q', '--quiet_text_graphics', action='store_true', dest='quiet_graphics',
                      help='Generate minimal output and no graphics', default=False)
    parser.add_option('-g', '--ghosts', dest='ghost',
                      help=default('the ghost agent TYPE in the ghostAgents module to use'),
                      metavar='TYPE', default='RandomGhost')
    parser.add_option('-k', '--num_ghosts', type='int', dest='num_ghosts',
                      help=default('The maximum number of ghosts to use'), default=4)
    parser.add_option('-z', '--zoom', type='float', dest='zoom',
                      help=default('Zoom the size of the graphics window'), default=1.0)
    parser.add_option('-f', '--fix_random_seed', action='store_true', dest='fix_random_seed',
                      help='Fixes the random seed to always play the same game', default=False)
    parser.add_option('-r', '--record_actions', action='store_true', dest='record',
                      help='Writes game histories to a file (named by the time they were played)', default=False)
    parser.add_option('--replay', dest='game_to_replay',
                      help='A recorded game file (pickle) to replay', default=None)
    parser.add_option('-a', '--agent_args', dest='agent_args',
                      help='Comma separated values sent to agent. e.g. "opt1=val1,opt2,opt3=val3"')
    parser.add_option('-x', '--num_training', dest='num_training', type='int',
                      help=default('How many episodes are training (suppresses output)'), default=0)
    parser.add_option('--frame_time', dest='frame_time', type='float',
                      help=default('Time to delay between frames; <0 means keyboard'), default=0.1)
    parser.add_option('-c', '--catch_exceptions', action='store_true', dest='catch_exceptions',
                      help='Turns on exception handling and timeouts during games', default=False)
    parser.add_option('--timeout', dest='timeout', type='int',
                      help=default('Maximum length of time an agent can spend computing in a single game'), default=30)

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    cmd_args: dict[str, Any] = dict()

    # Fix the random seed
    if options.fix_random_seed: random.seed('cs188')

    # Choose a layout
    cmd_args['layout'] = layout_module.get_layout(options.layout)
    if cmd_args['layout'] is None: raise Exception("The layout " + options.layout + " cannot be found")

    # Choose a Pacman agent
    no_keyboard = options.game_to_replay is None and (options.text_graphics or options.quiet_graphics)
    pacman_type = load_agent(options.pacman, no_keyboard)
    agent_opts = parse_agent_args(options.agent_args)
    if options.num_training > 0:
        cmd_args['num_training'] = options.num_training
        if 'numTraining' not in agent_opts: agent_opts['numTraining'] = options.num_training
    pacman_agent = pacman_type(**agent_opts)  # Instantiate Pacman with agent_opts
    cmd_args['pacman'] = pacman_agent

    # Don't display training games
    if 'numTrain' in agent_opts:
        options.num_quiet = int(agent_opts['numTrain'])
        options.num_ignore = int(agent_opts['numTrain'])

    # Choose a ghost agent
    ghost_type = load_agent(options.ghost, no_keyboard)
    cmd_args['ghosts'] = [ghost_type(i + 1) for i in range(options.num_ghosts)]

    # Choose a display format
    if options.quiet_graphics:
        from pacman.display import text_display
        cmd_args['display'] = text_display.NullGraphics()
    elif options.text_graphics:
        from pacman.display import text_display
        text_display.SLEEP_TIME = options.frame_time
        cmd_args['display'] = text_display.PacmanGraphics()
    else:
        from pacman.display import graphics_display
        cmd_args['display'] = graphics_display.PacmanGraphics(options.zoom, frame_time=options.frame_time)
    cmd_args['num_games'] = options.num_games
    cmd_args['record'] = options.record
    cmd_args['catch_exceptions'] = options.catch_exceptions
    cmd_args['timeout'] = options.timeout

    # Special case: recorded games don't use the run_games method or args structure
    if options.game_to_replay is not None:
        print('Replaying recorded game %s.' % options.game_to_replay)
        import pickle
        f = open(options.game_to_replay, 'rb')
        try:
            recorded = pickle.load(f)
        finally:
            f.close()
        recorded['display'] = cmd_args['display']
        replay_game(**recorded)
        sys.exit(0)

    return cmd_args


def load_agent(pacman: str, nographics: bool) -> type:
    # Scan the pacman.agents subpackage for agent classes
    import importlib
    import pkgutil
    import pacman.agents as agents_pkg

    for importer, modname, ispkg in pkgutil.iter_modules(agents_pkg.__path__):
        if not modname.endswith('agents'):
            continue
        try:
            module = importlib.import_module(f'pacman.agents.{modname}')
        except ImportError:
            continue
        if pacman in dir(module):
            if nographics and modname == 'keyboard_agents':
                raise Exception('Using the keyboard requires graphics (not text display)')
            return getattr(module, pacman)
    raise Exception('The agent ' + pacman + ' is not specified in any agents module.')


def replay_game(layout: Layout, actions: list[tuple[int, str]], display: Any) -> None:
    from pacman.agents import pacman_agents, ghost_agents
    rules = ClassicGameRules()
    agents = [pacman_agents.GreedyAgent()] + [ghost_agents.RandomGhost(i + 1) for i in range(layout.get_num_ghosts())]
    game = rules.new_game(layout, agents[0], agents[1:], display)
    state = game.state
    display.initialize(state.data)

    for action in actions:
        # Execute the action
        state = state.generate_successor(*action)
        # Change the display
        display.update(state.data)
        # Allow for game specific conditions (winning, losing, etc.)
        rules.process(state, game)

    display.finish()


def run_games(layout: Layout, pacman: Agent, ghosts: list[Agent], display: Any, num_games: int, record: bool,
              num_training: int = 0, catch_exceptions: bool = False, timeout: int = 30) -> list[Game]:
    import __main__
    __main__.__dict__['_display'] = display

    rules = ClassicGameRules(timeout)
    games: list[Game] = []

    for i in range(num_games):
        be_quiet = i < num_training
        if be_quiet:
            # Suppress output and graphics
            from pacman.display import text_display
            game_display = text_display.NullGraphics()
            rules.quiet = True
        else:
            game_display = display
            rules.quiet = False
        game = rules.new_game(layout, pacman, ghosts, game_display, be_quiet, catch_exceptions)
        game.run()
        if not be_quiet: games.append(game)

        if record:
            import time as time_mod
            import pickle
            fname = ('recorded-game-%d' % (i + 1)) + '-'.join([str(t) for t in time_mod.localtime()[1:6]])
            f = open(fname, 'wb')
            components = {'layout': layout, 'actions': game.move_history}
            pickle.dump(components, f)
            f.close()

    if (num_games - num_training) > 0:
        scores = [game.state.get_score() for game in games]
        wins = [game.state.is_win() for game in games]
        win_rate = wins.count(True) / float(len(wins))
        print('Average Score:', sum(scores) / float(len(scores)))
        print('Scores:       ', ', '.join([str(score) for score in scores]))
        print('Win Rate:      %d/%d (%.2f)' % (wins.count(True), len(wins), win_rate))
        print('Record:       ', ', '.join([['Loss', 'Win'][int(w)] for w in wins]))

    return games


def main() -> None:
    """
    The main function called when engine.py is run
    from the command line:

    > python -m pacman.engine

    See the usage string for more details.

    > python -m pacman.engine --help
    """
    import os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    run_args = read_command(sys.argv[1:])  # Get game components based on input
    run_games(**run_args)


if __name__ == '__main__':
    main()
