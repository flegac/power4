from math import sqrt, log2

from src.games.state.State import State


class Node:
    def __init__(self, state: State, parent=None) -> None:
        self.parent = parent
        self.depth = parent.depth + 1 if parent else 0

        self.state = state
        self.score = 0
        self.games = 0
        self._children = None

    def best_move(self):
        side_factor = 1 if self.state.board.current_turn % 2 == 0 else -1
        children = self.children()

        def action_value(action):
            return side_factor * children[action].exploitation_score()

        return max(children.keys(), key=action_value)

    def children(self):
        if self._children is None:
            next_states = {a: self.state.copy().next(a) for a in self.state.actions()}
            self._children = {a: Node(state=next_states[a], parent=self) for a in next_states}
        return self._children

    def exploitation_score(self):
        if self.state.is_terminal:
            return self.state.terminal_result
        exploitation = self.score / self.games if self.games != 0 else 0
        return exploitation

    def exploration_score(self):
        return sqrt(log2(self.parent.games) / max(self.games, 1)) if self.parent else 0

    def __repr__(self) -> str:
        parent_games = self.parent.games if self.parent else 0
        return '[depth={}, score={}, exploration={} ({}/{})]'.format(self.depth,
                                                                     int(100 * self.exploitation_score()),
                                                                     int(100 * self.exploration_score()),
                                                                     self.games,
                                                                     parent_games)
