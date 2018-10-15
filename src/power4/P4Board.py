import numpy as np

from src.games.state.Board import Board


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

    def id(self):
        return str(self.grid)

    def copy(self):
        return P4Board(np.copy(self.grid), np.copy(self.heights), self.current_turn)

    def __str__(self) -> str:
        return str(np.rot90(self.grid)).replace('0', '.').replace('1', 'O').replace('2', 'X')

    def to_training(self):
        p1_func = np.vectorize(lambda x: 1 if (x % 3) == 1 else 0)
        p1 = p1_func(self.grid)

        p2_func = np.vectorize(lambda x: 1 if (x % 3) == 2 else 0)
        p2 = p2_func(self.grid)
        return np.array([p1, p2], dtype=np.float32)
