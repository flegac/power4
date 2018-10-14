import pprint
import random

from src.games.state.State import State
import time

from src.games.util.MctsNode import MctsNode


class Mcts:
    def __init__(self, initial_state: State = None, root_node: MctsNode = None) -> None:
        if root_node is None:
            root_node = MctsNode(initial_state)
        self.root_node = root_node
        self.tree = {self.root_node}
        self.depth = 0

        self.select_time = 0
        self.expand_time = 0
        self.simulate_time = 0
        self.backpropagation_time = 0
        self.run_time = 0

    def run(self, n):
        start = time.time()
        for i in range(0, n):
            initial = self.select(self.root_node)
            expansion = self.expand(initial)
            reward = self.simulate(expansion)
            self.backpropagation(expansion, reward)
        delta = time.time() - start
        self.run_time += delta
        return delta

    def select(self, initial_node: MctsNode) -> MctsNode:
        start = time.time()

        def side_exploitation_score(node: MctsNode):
            side = -1 if node.state.board.current_turn % 2 == 0 else 1
            return side * node.exploitation_score()

        def node_score(node: MctsNode):
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

    def expand(self, node: MctsNode):
        start = time.time()
        if node.state.is_terminal:
            self.expand_time += time.time() - start
            return node
        if node.games == 0:
            self._register_node(node)
            self.expand_time += time.time() - start
            return node
        try:
            children = node.children()
            child = random.choice(list(children.values()))
        except Exception as e:
            raise e
        self._register_node(child)
        self.expand_time += time.time() - start
        return child

    def _register_node(self, node: MctsNode):
        self.tree.add(node)
        self.depth = max(self.depth, node.depth)

    def simulate(self, node: MctsNode):
        start = time.time()
        if node.state.is_terminal:
            self.simulate_time += time.time() - start
            return node.state.terminal_result
        current = node.state.copy()
        while not current.is_terminal:
            action = random.choice(current.actions())
            current.apply(action)
        self.simulate_time += time.time() - start
        return current.terminal_result

    def backpropagation(self, node: MctsNode, score):
        start = time.time()
        depth = 0
        while node is not self.root_node:
            node.score += score
            node.games += 1
            node = node.parent
            depth += 1
        node.score += score
        node.games += 1
        self.backpropagation_time += time.time() - start

    def stats(self):
        print(self.tree_stats())
        print(self.thinking_stats())
        print(self.time_stats())

    def tree_stats(self):
        return pprint.pformat({
            'tree final size': len(self.tree),
            'tree depth': self.depth,
            'tree root playouts': self.root_node.games
        }, indent=2)

    def thinking_stats(self):
        return pprint.pformat(self.root_node.children())

    def time_stats(self):
        return 'time:\n select={}\n expand={}\n simulate={}\n backpropagation={}\n total={}'.format(
            self.select_time,
            self.expand_time,
            self.simulate_time,
            self.backpropagation_time,
            self.run_time
        )
