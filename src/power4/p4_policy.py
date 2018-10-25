import numpy as np

from src.deep.MyModel import MyModel
from src.games.Game import Game
from src.games.Policy import Policy, MctsPolicy2
from src.games.state.State import State
from src.power4.P4Board import P4Board
from src.power4.P4Rules import P4Rules


class P4Policy(Policy):

    def __init__(self, model: MyModel) -> None:
        super().__init__(model.name)
        self.model = model

    def play(self, state: State) -> int:
        actions = state.actions()

        moves = {a: _gen_sample_input(state, state.next(a)) for a in actions}
        side = 1 if state.board.current_turn % 2 == 0 else -1

        evaluations = {x: side * self.model.predict(moves[x]) for x in moves}
        print(evaluations)

        a = -1
        s = -side * 1000
        for x in evaluations:
            if evaluations[x] > s:
                s = evaluations[x]
                a = x

        return x


def p4_board_preparation(board: P4Board):
    p1_func = np.vectorize(lambda x: 1.0 if (x % 3) == 1 else 0)
    p1 = p1_func(board.grid)
    p2_func = np.vectorize(lambda x: -1.0 if (x % 3) == 2 else 0)
    p2 = p2_func(board.grid)
    return p1, p2


def _gen_sample_input(state1: State, state2: State) -> np.ndarray:
    p1, p2 = p4_board_preparation(state1.board)
    p3, p4 = p4_board_preparation(state2.board)
    return np.array([p1, p2, p3, p4], dtype=np.float32)


policy = P4Policy(MyModel.load('p4_model_v1_final'))

computer_mcts = MctsPolicy2(n=2000, after_play=lambda mcts: print(mcts))

state = P4Rules.initial_state()

history = []
game = Game(P4Rules.initial_state(), policy, computer_mcts)
game.run(on_state_change=lambda s: print(str(s)))
