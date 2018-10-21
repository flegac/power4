import numpy as np

from src.power4.P4Board import P4Board
from src.power4.p4_sample_generator import SampleGenerator


def p4_board_preparation(board: P4Board):
    p1_func = np.vectorize(lambda x: 1.0 if (x % 3) == 1 else 0)
    p1 = p1_func(board.grid)

    p2_func = np.vectorize(lambda x: -1.0 if (x % 3) == 2 else 0)
    p2 = p2_func(board.grid)

    # side = 1 if board.current_turn % 2 == 0 else -1
    # initiative_side = np.full((P4Board.GRID_WIDTH, P4Board.GRID_HEIGHT), side)

    return p1, p2


SampleGenerator('p4_', board_preparation=p4_board_preparation) \
    .gen_batch(batch_size=1000, batch_number=1000)

print('done !')
