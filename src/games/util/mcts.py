import random

from src.games.state.State import State
import time

from src.games.util.mcts_node import Node


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

    def run(self, n):
        for i in range(0, n):
            initial = self.select(self.root)
            expansion = self.expand(initial)
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
            candidates = current.children().values()

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
        try:
            children = node.children()
            child = random.choice(list(children.values()))
        except Exception as e:
            raise e
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

    def stats(self):
        print('tree final size: {}'.format(len(self.tree)))
        print('games : ', self.games)
        print('scores :')
        for action in self.root.children():
            print('{} : {}'.format(action, self.root.children()[action]))
        template = 'time:\n select={}\n expand={}\n simulate={}\n backpropagation={}\n'
        print(template.format(
            self.select_time,
            self.expand_time,
            self.simulate_time,
            self.backpropagation_time))
