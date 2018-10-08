from src.games.state.Board import Board
from src.games.state.Rules import Rules


class State:
    def __init__(self, rules: Rules, board: Board) -> None:
        self.rules = rules
        self.board = board
        self._data = {}
        self.is_terminal = False
        self.terminal_result = None

    def actions(self) -> []:
        return self.rules.actions(self)

    def next(self, action):
        return self.rules.next(self, action)

    def undo(self, last_action):
        return self.rules.undo(self, last_action)

    def copy(self):
        state = State(self.rules, self.board.copy())
        state.is_terminal = self.is_terminal
        state.terminal_result = self.terminal_result
        return state

    def get(self, attr: str = None):
        if attr:
            return self._data[attr]
        return self._data

    def set(self, attr: str = None, value=None, state: dict = None):
        if attr and value:
            self._data[attr] = value
        if state:
            self._data = state

    def __str__(self) -> str:
        return str(self.board)
