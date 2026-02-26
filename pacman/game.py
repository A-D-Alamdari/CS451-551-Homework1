"""
game.py
-------
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
from typing import Any, Optional
from pacman.util import raise_not_defined, nearest_point, TimeoutFunction, TimeoutFunctionException
import time
import traceback
import sys
import io


#######################
# Parts worth reading #
#######################

class Agent:
    """
    An agent must define a get_action method, but may also define the
    following methods which will be called if they exist:

    def register_initial_state(self, state): # inspects the starting state
    """

    def __init__(self, index: int = 0) -> None:
        self.index: int = index

    def get_action(self, state: Any) -> str:
        """
        The Agent will receive a GameState (from either {pacman, capture, sonar}.py) and
        must return an action from Directions.{North, South, East, West, Stop}
        """
        raise_not_defined()
        raise NotImplementedError  # unreachable; raise_not_defined() always exits


class Directions:
    NORTH: str = 'North'
    SOUTH: str = 'South'
    EAST: str = 'East'
    WEST: str = 'West'
    STOP: str = 'Stop'

    LEFT: dict[str, str] = {NORTH: WEST,
                            SOUTH: EAST,
                            EAST: NORTH,
                            WEST: SOUTH,
                            STOP: STOP}

    RIGHT: dict[str, str] = dict([(y, x) for x, y in LEFT.items()])

    REVERSE: dict[str, str] = {NORTH: SOUTH,
                               SOUTH: NORTH,
                               EAST: WEST,
                               WEST: EAST,
                               STOP: STOP}


class Configuration:
    """
    A Configuration holds the (x,y) coordinate of a character, along with its
    traveling direction.

    The convention for positions, like a graph, is that (0,0) is the lower left corner, x increases
    horizontally and y increases vertically.  Therefore, north is the direction of increasing y, or (0,1).
    """

    def __init__(self, pos: tuple[float, float], direction: str) -> None:
        self.pos: tuple[float, float] = pos
        self.direction: str = direction

    def get_position(self) -> tuple[float, float]:
        return self.pos

    def get_direction(self) -> str:
        return self.direction

    def is_integer(self) -> bool:
        x, y = self.pos
        return x == int(x) and y == int(y)

    def __eq__(self, other: object) -> bool:
        if other is None: return False
        if not isinstance(other, Configuration): return NotImplemented
        return self.pos == other.pos and self.direction == other.direction

    def __hash__(self) -> int:
        x = hash(self.pos)
        y = hash(self.direction)
        return hash(x + 13 * y)

    def __str__(self) -> str:
        return "(x,y)=" + str(self.pos) + ", " + str(self.direction)

    def generate_successor(self, vector: tuple[float, float]) -> Configuration:
        """
        Generates a new configuration reached by translating the current
        configuration by the action vector.  This is a low-level call and does
        not attempt to respect the legality of the movement.

        Actions are movement vectors.
        """
        x, y = self.pos
        dx, dy = vector
        direction = Actions.vector_to_direction(vector)
        if direction == Directions.STOP:
            direction = self.direction  # There is no stop direction
        return Configuration((x + dx, y + dy), direction)


class AgentState:
    """
    AgentStates hold the state of an agent (configuration, speed, scared, etc.).
    """

    def __init__(self, start_configuration: Configuration, is_pacman: bool) -> None:
        self.start: Configuration = start_configuration
        self.configuration: Configuration = start_configuration
        self.is_pacman: bool = is_pacman
        self.scared_timer: int = 0
        self.num_carrying: int = 0
        self.num_returned: int = 0

    def __str__(self) -> str:
        if self.is_pacman:
            return "Pacman: " + str(self.configuration)
        else:
            return "Ghost: " + str(self.configuration)

    def __eq__(self, other: object) -> bool:
        if other is None:
            return False
        if not isinstance(other, AgentState):
            return NotImplemented
        return self.configuration == other.configuration and self.scared_timer == other.scared_timer

    def __hash__(self) -> int:
        return hash(hash(self.configuration) + 13 * hash(self.scared_timer))

    def copy(self) -> AgentState:
        state = AgentState(self.start, self.is_pacman)
        state.configuration = self.configuration
        state.scared_timer = self.scared_timer
        state.num_carrying = self.num_carrying
        state.num_returned = self.num_returned
        return state

    def get_position(self) -> Optional[tuple[float, float]]:
        if self.configuration is None: return None
        return self.configuration.get_position()

    def get_direction(self) -> str:
        return self.configuration.get_direction()


class Grid:
    """
    A 2-dimensional array of objects backed by a list of lists.  Data is accessed
    via grid[x][y] where (x,y) are positions on a Pacman map with x horizontal,
    y vertical and the origin (0,0) in the bottom left corner.

    The __str__ method constructs an output that is oriented like a pacman board.
    """

    def __init__(self, width: int, height: int, initial_value: Any = False,
                 bit_representation: Optional[tuple[int, ...]] = None) -> None:
        if initial_value not in [False, True]: raise Exception('Grids can only contain booleans')
        self.CELLS_PER_INT: int = 30

        self.width: int = width
        self.height: int = height
        self.data: list[list[Any]] = [[initial_value for _y in range(height)] for _x in range(width)]
        if bit_representation:
            self._unpack_bits(bit_representation)

    def __getitem__(self, i: int) -> list[Any]:
        return self.data[i]

    def __setitem__(self, key: int, item: list[Any]) -> None:
        self.data[key] = item

    def __str__(self) -> str:
        out = [[str(self.data[x][y])[0] for x in range(self.width)] for y in range(self.height)]
        out.reverse()
        return '\n'.join([''.join(x) for x in out])

    def __eq__(self, other: object) -> bool:
        if other is None: return False
        if not isinstance(other, Grid): return NotImplemented
        return self.data == other.data

    def __hash__(self) -> int:
        # return hash(str(self))
        base = 1
        h = 0
        for l in self.data:
            for i in l:
                if i:
                    h += base
                base *= 2
        return hash(h)

    def copy(self) -> Grid:
        g = Grid(self.width, self.height)
        g.data = [x[:] for x in self.data]
        return g

    def deep_copy(self) -> Grid:
        return self.copy()

    def shallow_copy(self) -> Grid:
        g = Grid(self.width, self.height)
        g.data = self.data
        return g

    def count(self, item: Any = True) -> int:
        return sum([x.count(item) for x in self.data])

    def as_list(self, key: Any = True) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for x in range(self.width):
            for y in range(self.height):
                if self[x][y] == key: result.append((x, y))
        return result

    def pack_bits(self) -> tuple[int, ...]:
        """
        Returns an efficient int list representation

        (width, height, bitPackedInts...)
        """
        bits: list[int] = [self.width, self.height]
        current_int = 0
        for i in range(self.height * self.width):
            bit = self.CELLS_PER_INT - (i % self.CELLS_PER_INT) - 1
            x, y = self._cell_index_to_position(i)
            if self[x][y]:
                current_int += 2 ** bit
            if (i + 1) % self.CELLS_PER_INT == 0:
                bits.append(current_int)
                current_int = 0
        bits.append(current_int)
        return tuple(bits)

    def _cell_index_to_position(self, index: int) -> tuple[int, int]:
        x = index // self.height
        y = index % self.height
        return x, y

    def _unpack_bits(self, bits: tuple[int, ...]) -> None:
        """
        Fills in data from a bit-level representation
        """
        cell = 0
        for packed in bits:
            for bit in self._unpack_int(packed, self.CELLS_PER_INT):
                if cell == self.width * self.height: break
                x, y = self._cell_index_to_position(cell)
                self[x][y] = bit
                cell += 1

    def _unpack_int(self, packed: int, size: int) -> list[bool]:
        bools: list[bool] = []
        if packed < 0: raise ValueError("must be a positive integer")
        for _i in range(size):
            n = 2 ** (self.CELLS_PER_INT - _i - 1)
            if packed >= n:
                bools.append(True)
                packed -= n
            else:
                bools.append(False)
        return bools


def reconstitute_grid(bit_rep: Any) -> Any:
    if not isinstance(bit_rep, tuple):
        return bit_rep
    width, height = bit_rep[:2]
    return Grid(width, height, bit_representation=bit_rep[2:])


####################################
# Parts you shouldn't have to read #
####################################

class Actions:
    """
    A collection of static methods for manipulating move actions.
    """
    # Directions
    _directions: dict[str, tuple[int, int]] = {
        Directions.NORTH: (0, 1),
        Directions.SOUTH: (0, -1),
        Directions.EAST: (1, 0),
        Directions.WEST: (-1, 0),
        Directions.STOP: (0, 0),
    }

    _directions_as_list = _directions.items()

    TOLERANCE: float = .001

    def reverse_direction(action: str) -> str:
        if action == Directions.NORTH:
            return Directions.SOUTH
        if action == Directions.SOUTH:
            return Directions.NORTH
        if action == Directions.EAST:
            return Directions.WEST
        if action == Directions.WEST:
            return Directions.EAST
        return action

    reverse_direction = staticmethod(reverse_direction)

    def vector_to_direction(vector: tuple[float, float]) -> str:
        dx, dy = vector
        if dy > 0:
            return Directions.NORTH
        if dy < 0:
            return Directions.SOUTH
        if dx < 0:
            return Directions.WEST
        if dx > 0:
            return Directions.EAST
        return Directions.STOP

    vector_to_direction = staticmethod(vector_to_direction)

    def direction_to_vector(direction: str, speed: float = 1.0) -> tuple[float, float]:
        dx, dy = Actions._directions[direction]
        return dx * speed, dy * speed

    direction_to_vector = staticmethod(direction_to_vector)

    def get_possible_actions(config: Configuration, walls: Grid) -> list[str]:
        possible: list[str] = []
        x, y = config.pos
        x_int, y_int = int(x + 0.5), int(y + 0.5)

        # In between grid points, all agents must continue straight
        if abs(x - x_int) + abs(y - y_int) > Actions.TOLERANCE:
            return [config.get_direction()]

        for direction, vec in Actions._directions_as_list:
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if not walls[next_x][next_y]: possible.append(direction)

        return possible

    get_possible_actions = staticmethod(get_possible_actions)

    def get_legal_neighbors(position: tuple[float, float], walls: Grid) -> list[tuple[int, int]]:
        x, y = position
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        neighbors: list[tuple[int, int]] = []
        for direction, vec in Actions._directions_as_list:
            dx, dy = vec
            next_x = x_int + dx
            if next_x < 0 or next_x == walls.width: continue
            next_y = y_int + dy
            if next_y < 0 or next_y == walls.height: continue
            if not walls[next_x][next_y]: neighbors.append((next_x, next_y))
        return neighbors

    get_legal_neighbors = staticmethod(get_legal_neighbors)

    def get_successor(position: tuple[float, float], action: str) -> tuple[float, float]:
        dx, dy = Actions.direction_to_vector(action)
        x, y = position
        return x + dx, y + dy

    get_successor = staticmethod(get_successor)


class GameStateData:
    """

    """

    def __init__(self, prev_state: Optional[GameStateData] = None) -> None:
        """
        Generates a new data packet by copying information from its predecessor.
        """
        if prev_state is not None:
            self.food: Grid = prev_state.food.shallow_copy()
            self.capsules: list[tuple[int, int]] = prev_state.capsules[:]
            self.agent_states: list[AgentState] = self.copy_agent_states(prev_state.agent_states)
            self.layout: Any = prev_state.layout
            self._eaten: list[bool] = prev_state._eaten
            self.score: int = prev_state.score

        self._food_eaten: Optional[tuple[int, int]] = None
        self._food_added: Optional[tuple[int, int]] = None
        self._capsule_eaten: Optional[tuple[int, int]] = None
        self._agent_moved: Optional[int] = None
        self._lose: bool = False
        self._win: bool = False
        self.score_change: int = 0

    def deep_copy(self) -> GameStateData:
        state = GameStateData(self)
        state.food = self.food.deep_copy()
        state.layout = self.layout.deep_copy()
        state._agent_moved = self._agent_moved
        state._food_eaten = self._food_eaten
        state._food_added = self._food_added
        state._capsule_eaten = self._capsule_eaten
        return state

    @staticmethod
    def copy_agent_states(agent_states: list[AgentState]) -> list[AgentState]:
        copied_states: list[AgentState] = []
        for agent_state in agent_states:
            copied_states.append(agent_state.copy())
        return copied_states

    def __eq__(self, other: object) -> bool:
        """
        Allows two states to be compared.
        """
        if other is None: return False
        if not isinstance(other, GameStateData): return NotImplemented
        # TODO Check for type of other
        if not self.agent_states == other.agent_states: return False
        if not self.food == other.food: return False
        if not self.capsules == other.capsules: return False
        if not self.score == other.score: return False
        return True

    def __hash__(self) -> int:
        """
        Allows states to be keys of dictionaries.
        """
        for _i, state in enumerate(self.agent_states):
            try:
                int(hash(state))
            except TypeError as e:
                print(e)
                # hash(state)
        return int((hash(tuple(self.agent_states)) + 13 * hash(self.food) + 113 * hash(tuple(self.capsules)) + 7 * hash(
            self.score)) % 1048575)

    def __str__(self) -> str:
        width, height = self.layout.width, self.layout.height
        grid_map = Grid(width, height)
        if isinstance(self.food, tuple):
            self.food = reconstitute_grid(self.food)
        for x in range(width):
            for y in range(height):
                food, walls = self.food, self.layout.walls
                grid_map[x][y] = self._food_wall_str(food[x][y], walls[x][y])

        for agent_state in self.agent_states:
            if agent_state is None: continue
            if agent_state.configuration is None: continue
            x, y = [int(i) for i in nearest_point(agent_state.configuration.pos)]
            agent_dir = agent_state.configuration.direction
            if agent_state.is_pacman:
                grid_map[x][y] = self._pac_str(agent_dir)
            else:
                grid_map[x][y] = self._ghost_str(agent_dir)

        for x, y in self.capsules:
            grid_map[x][y] = 'o'

        return str(grid_map) + ("\nScore: %d\n" % self.score)

    @staticmethod
    def _food_wall_str(has_food: bool, has_wall: bool) -> str:
        if has_food:
            return '.'
        elif has_wall:
            return '%'
        else:
            return ' '

    @staticmethod
    def _pac_str(direction: str) -> str:
        if direction == Directions.NORTH:
            return 'v'
        if direction == Directions.SOUTH:
            return '^'
        if direction == Directions.WEST:
            return '>'
        return '<'

    @staticmethod
    def _ghost_str(direction: str) -> str:
        if direction == Directions.NORTH:
            return 'M'
        if direction == Directions.SOUTH:
            return 'W'
        if direction == Directions.WEST:
            return '3'
        if direction == Directions.EAST:
            return 'E'
        return 'G'

    def initialize(self, layout: Any, num_ghost_agents: int) -> None:
        """
        Creates an initial game state from a layout array (see layout.py).
        """
        self.food = layout.food.copy()
        # self.capsules = []
        self.capsules = layout.capsules[:]
        self.layout = layout
        self.score = 0
        self.score_change = 0

        self.agent_states = []
        num_ghosts = 0
        for is_pacman, pos in layout.agent_positions:
            if not is_pacman:
                if num_ghosts == num_ghost_agents:
                    continue  # Max ghosts reached already
                else:
                    num_ghosts += 1
            self.agent_states.append(AgentState(Configuration(pos, Directions.STOP), is_pacman))
        self._eaten = [False for _a in self.agent_states]


try:
    import boinc

    _BOINC_ENABLED = True
except ImportError:
    boinc = None  # type: ignore[assignment]
    _BOINC_ENABLED = False

# Module-level stdout/stderr saved references used by Game.mute/unmute
OLD_STDOUT: Optional[Any] = None
OLD_STDERR: Optional[Any] = None


class Game:
    """
    The Game manages the control flow, soliciting actions from agents.
    """

    def __init__(self, agents: list[Agent], display: Any, rules: Any, starting_index: int = 0,
                 mute_agents: bool = False, catch_exceptions: bool = False) -> None:
        self.agent_crashed: bool = False
        self.agents: list[Agent] = agents
        self.display: Any = display
        self.rules: Any = rules
        self.starting_index: int = starting_index
        self.game_over: bool = False
        self.mute_agents: bool = mute_agents
        self.catch_exceptions: bool = catch_exceptions
        self.move_history: list[tuple[int, str]] = []
        self.total_agent_times: list[float] = [0 for _agent in agents]
        self.total_agent_time_warnings: list[int] = [0 for _agent in agents]
        self.agent_timeout: bool = False
        self.agent_output: list[io.StringIO] = [io.StringIO() for _agent in agents]
        self.state: Any = None
        self.num_moves: int = 0

    def get_progress(self) -> float:
        if self.game_over:
            return 1.0
        else:
            return self.rules.get_progress(self)

    def _agent_crash(self, agent_index: int, quiet: bool = False) -> None:
        """Helper method for handling agent crashes"""
        if not quiet: traceback.print_exc()
        self.game_over = True
        self.agent_crashed = True
        self.rules.agent_crash(self, agent_index)

    def mute(self, agent_index: int) -> None:
        if not self.mute_agents: return
        global OLD_STDOUT, OLD_STDERR
        OLD_STDOUT = sys.stdout
        OLD_STDERR = sys.stderr
        sys.stdout = self.agent_output[agent_index]
        sys.stderr = self.agent_output[agent_index]

    def unmute(self) -> None:
        if not self.mute_agents: return
        global OLD_STDOUT, OLD_STDERR
        # Revert stdout/stderr to originals
        sys.stdout = OLD_STDOUT  # type: ignore[assignment]
        sys.stderr = OLD_STDERR  # type: ignore[assignment]

    def run(self) -> None:
        """
        Main control loop for game play.
        """
        self.display.initialize(self.state.data)
        self.num_moves = 0

        ###self.display.initialize(self.state.makeObservation(1).data)
        # inform learning agents of the game start
        for i in range(len(self.agents)):
            agent = self.agents[i]
            if not agent:
                self.mute(i)
                # this is a null agent, meaning it failed to load
                # the other team wins
                print("Agent %d failed to load" % i, file=sys.stderr)
                self.unmute()
                self._agent_crash(i, quiet=True)
                return
            if hasattr(agent, 'register_initial_state'):
                self.mute(i)
                if self.catch_exceptions:
                    try:  # noqa: BLE001
                        timed_func = TimeoutFunction(getattr(agent, 'register_initial_state'),
                                                     int(self.rules.get_max_startup_time(i)))
                        try:
                            start_time = time.time()
                            timed_func(self.state.deep_copy())
                            time_taken = time.time() - start_time
                            self.total_agent_times[i] += time_taken
                        except TimeoutFunctionException:
                            print("Agent %d ran out of time on startup!" % i, file=sys.stderr)
                            self.unmute()
                            self.agent_timeout = True
                            self._agent_crash(i, quiet=True)
                            return
                    except Exception:  # noqa: BLE001
                        self._agent_crash(i, quiet=False)
                        self.unmute()
                        return
                else:
                    getattr(agent, 'register_initial_state')(self.state.deep_copy())
                ## TODO: could this exceed the total time
                self.unmute()

        agent_index = self.starting_index
        num_agents = len(self.agents)

        while not self.game_over:
            # Fetch the next agent
            agent = self.agents[agent_index]
            move_time: float = 0.0
            skip_action = False
            # Generate an observation of the state
            observation: Any = self.state.deep_copy()
            if hasattr(agent, 'observationFunction'):
                self.mute(agent_index)
                if self.catch_exceptions:
                    try:  # noqa: BLE001
                        timed_func = TimeoutFunction(getattr(agent, 'observationFunction'),
                                                     int(self.rules.get_move_timeout(agent_index)))
                        try:
                            start_time: float = time.time()
                            observation = timed_func(self.state.deep_copy())
                        except TimeoutFunctionException:
                            skip_action = True
                        move_time += time.time() - start_time
                        self.unmute()
                    except Exception:  # noqa: BLE001
                        self._agent_crash(agent_index, quiet=False)
                        self.unmute()
                        return
                else:
                    observation = getattr(agent, 'observationFunction')(self.state.deep_copy())
                self.unmute()

            # Solicit an action
            action: str = Directions.STOP
            self.mute(agent_index)
            if self.catch_exceptions:
                try:  # noqa: BLE001
                    timed_func = TimeoutFunction(agent.get_action,
                                                 int(self.rules.get_move_timeout(agent_index)) - int(move_time))
                    try:
                        start_time = time.time()
                        if skip_action:
                            raise TimeoutFunctionException()
                        action = timed_func(observation)
                    except TimeoutFunctionException:
                        print("Agent %d timed out on a single move!" % agent_index, file=sys.stderr)
                        self.agent_timeout = True
                        self._agent_crash(agent_index, quiet=True)
                        self.unmute()
                        return

                    move_time += time.time() - start_time

                    if move_time > self.rules.get_move_warning_time(agent_index):
                        self.total_agent_time_warnings[agent_index] += 1
                        print("Agent %d took too long to make a move! This is warning %d" % (agent_index,
                                                                                             self.total_agent_time_warnings[
                                                                                                 agent_index]),
                              file=sys.stderr)
                        if self.total_agent_time_warnings[agent_index] > self.rules.get_max_time_warnings(agent_index):
                            print("Agent %d exceeded the maximum number of warnings: %d" % (agent_index,
                                                                                            self.total_agent_time_warnings[
                                                                                                agent_index]),
                                  file=sys.stderr)
                            self.agent_timeout = True
                            self._agent_crash(agent_index, quiet=True)
                            self.unmute()
                            return

                    self.total_agent_times[agent_index] += move_time
                    # print("Agent: %d, time: %f, total: %f" % (agent_index, move_time, self.total_agent_times[agent_index]))
                    if self.total_agent_times[agent_index] > self.rules.get_max_total_time(agent_index):
                        print("Agent %d ran out of time! (time: %1.2f)" % (agent_index,
                                                                           self.total_agent_times[agent_index]),
                              file=sys.stderr)
                        self.agent_timeout = True
                        self._agent_crash(agent_index, quiet=True)
                        self.unmute()
                        return
                    self.unmute()
                except Exception:  # noqa: BLE001
                    self._agent_crash(agent_index)
                    self.unmute()
                    return
            else:
                action = agent.get_action(observation)
            self.unmute()

            # Execute the action
            self.move_history.append((agent_index, action))
            if self.catch_exceptions:
                try:  # noqa: BLE001
                    self.state = self.state.generate_successor(agent_index, action)
                except Exception:  # noqa: BLE001
                    self.mute(agent_index)
                    self._agent_crash(agent_index)
                    self.unmute()
                    return
            else:
                self.state = self.state.generate_successor(agent_index, action)

            # Change the display
            self.display.update(self.state.data)
            ###idx = agent_index - agent_index % 2 + 1
            ###self.display.update( self.state.makeObservation(idx).data )

            # Allow for game specific conditions (winning, losing, etc.)
            self.rules.process(self.state, self)
            # Track progress
            if agent_index == num_agents + 1: self.num_moves += 1
            # Next agent
            agent_index = (agent_index + 1) % num_agents

            if _BOINC_ENABLED:
                boinc.set_fraction_done(self.get_progress())

        # inform a learning agent of the game result
        for agent_index, agent in enumerate(self.agents):
            if hasattr(agent, 'final'):
                try:  # noqa: BLE001
                    self.mute(agent_index)
                    getattr(agent, 'final')(self.state)
                    self.unmute()
                except Exception:  # noqa: BLE001
                    if not self.catch_exceptions: raise
                    self._agent_crash(agent_index)
                    self.unmute()
                    return
        self.display.finish()
