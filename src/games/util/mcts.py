from math import *
import random

from src.games.state.State import State
import time


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
        return max(self.children(), key=lambda x: side_factor * x.exploitation_score())

    def children(self):
        if self._children is None:
            next_states = [self.state.copy().next(a) for a in self.state.actions()]
            self._children = [Node(child, parent=self) for child in next_states]
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


class MCTS:
    def __init__(self, state: State = None, node: Node = None) -> None:
        if node is None:
            node = Node(state)
        self.root = node
        self.tree = {self.root}
        self.games = [0] * 50

        self.select_time = 0
        self.expand_time = 0
        self.simulate_time = 0

        self.backpropagation_time = 0

    def run(self, n, m=1):
        for i in range(0, n):
            initial = self.select(self.root)
            expansion = self.expand(initial)
            for k in range(0, m):
                reward = self.simulate(expansion)
                self.backpropagation(expansion, reward)

    def select(self, initial_node: Node) -> Node:
        start = time.time()

        def side_exploitation_score(node: Node):
            side = -1 if node.state.board.current_turn % 2 == 0 else 1
            return side * node.exploitation_score()

        def node_score(node: Node):
            return side_exploitation_score(node) + node.exploration_score()

        current = initial_node
        while current.games > 0 and not current.state.is_terminal:
            candidates = current.children()

            next_current = max(candidates, key=side_exploitation_score)
            if next_current.state.is_terminal:
                break

            next_current = max(candidates, key=node_score)
            if next_current == current:
                break
            current = next_current
        self.select_time += time.time() - start
        return current

    def expand(self, node: Node):
        start = time.time()
        if node.state.is_terminal:
            self.expand_time += time.time() - start
            return node
        if node.games == 0:
            self.tree.add(node)
            self.expand_time += time.time() - start
            return node
        child = random.choice(node.children())
        self.tree.add(child)
        self.expand_time += time.time() - start
        return child

    def simulate(self, node: Node):
        start = time.time()
        if node.state.is_terminal:
            self.simulate_time += time.time() - start
            return node.state.terminal_result
        current = node.state.copy()
        while not current.is_terminal:
            action = random.choice(current.actions())
            current = current.next(action)
        self.simulate_time += time.time() - start
        return current.terminal_result

    def backpropagation(self, node: Node, score):
        start = time.time()
        depth = 0
        while node is not self.root:
            node.score += score
            node.games += 1
            node = node.parent
            self.games[depth] += 1
            depth += 1
        node.score += score
        node.games += 1
        self.games[depth] += 1
        self.backpropagation_time += time.time() - start
