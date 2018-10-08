import itertools

import numpy as np
from numpy import linalg as LA
from src.games.Evaluator import Evaluator
from src.games.state.Board import Board
from src.games.state.Rules import Rules
from src.games.state.State import State


class P4Board(Board):
    GRID_WIDTH = 7
    GRID_HEIGHT = 6

    def __init__(self, grid=None, heights=None, current_turn=0) -> None:
        super().__init__()
        if grid is None:
            grid = np.zeros((P4Board.GRID_WIDTH, P4Board.GRID_HEIGHT), dtype=np.int8)
        if heights is None:
            heights = np.zeros(P4Board.GRID_WIDTH, dtype=np.int8)
        self.grid = grid
        self.heights = heights
        self.current_turn = current_turn

    def copy(self):
        return P4Board(np.copy(self.grid), np.copy(self.heights), self.current_turn)

    def __str__(self) -> str:
        return str(np.rot90(self.grid)).replace('0', '.').replace('1', 'O').replace('2', 'X')


class P4Rules(Rules):

    @staticmethod
    def start():
        return State(P4Rules(), P4Board())

    @staticmethod
    def actions(state) -> []:
        actions = filter(P4Rules.is_valid(state), range(0, P4Board.GRID_WIDTH))
        return list(actions)

    @staticmethod
    def next(state, action):
        if not P4Rules.is_valid(state)(action):
            raise ValueError("invalid play : {}".format(action))
        height = state.board.heights[action]
        state.board.grid[action, height] = P4Rules.current_player(state)
        state.board.heights[action] += 1
        state.board.current_turn += 1

        P4Rules._update_terminal_state(state, action, height)
        return state

    @staticmethod
    def is_valid(state):
        return lambda column_id: state.board.heights[column_id] < P4Board.GRID_HEIGHT

    @staticmethod
    def current_player(state):
        current_player = 1 + state.board.current_turn % 2
        return current_player

    @staticmethod
    def _update_terminal_state(state, i, j):
        P4Rules._check_victory(state, i, j)
        # P4Rules._check_victory2(state, i, j)
        if not state.is_terminal:
            P4Rules._check_full_grid(state)

    @staticmethod
    def _check_full_grid(state):
        for x in state.board.heights:
            if x < P4Board.GRID_HEIGHT - 1:
                return
        state.is_terminal = True
        state.terminal_result = 0

    @staticmethod
    def _check_victory(state, i, j):
        grid = state.board.grid
        row = grid[i, :]
        line = grid[:, j]
        d1 = np.diag(grid, k=j - i)
        d2 = np.diag(np.rot90(grid), k=1 - grid.shape[1] + j + i)
        if P4Rules._check_line(row) or P4Rules._check_line(line) \
                or P4Rules._check_line(d1) or P4Rules._check_line(d2):
            state.is_terminal = True
            state.terminal_result = 2 * (state.board.current_turn % 2) - 1

    @staticmethod
    def _check_line(line):
        for g, v in itertools.groupby(line):
            groups = list(v)
            if len(groups) >= 4 and groups[0] != 0:
                return True
        return False

    @staticmethod
    def _check_victory2(state, i, j):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if P4Rules.is_power4(state, i, j, dx, dy):
                    state.is_terminal = True
                    state.terminal_result = 2 * (state.board.current_turn % 2) - 1
                    return

    @staticmethod
    def is_power4(state, x, y, dx, dy):
        color = state.board.grid[x, y]
        for i in range(1, 4):
            try:
                if state.board.grid[x + i * dx, y + i * dy] != color:
                    return False
            except:
                return False
        return True


class EvaluatorP4(Evaluator):

    def __init__(self) -> None:
        super().__init__(P4Board.GRID_WIDTH, P4Board.GRID_HEIGHT, brain_depth=10)

    def eval(self, state: State) -> [float]:
        choices = self.bias + np.dot(state.board.grid, self.weights)
        choices = [LA.norm(x) for x in choices]
        return choices
