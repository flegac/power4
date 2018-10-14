from src.games.state.Board import Board
from src.games.state.Rules import Rules


class State:
    def __init__(self, rules: Rules, board: Board) -> None:
        self.rules = rules
        self.board = board
        self.is_terminal = False
        self.terminal_result = None

    def actions(self) -> []:
        return self.rules.actions(self)

    def apply(self, action):
        return self.rules.apply(self, action)

    def next(self, action):
        return self.copy().apply(action)

    def copy(self):
        state = State(self.rules, self.board.copy())
        state.is_terminal = self.is_terminal
        state.terminal_result = self.terminal_result
        return state

    def __str__(self) -> str:
        return str(self.board)
