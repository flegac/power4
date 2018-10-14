import pprint
import random

from src.games.mcts.TreeStats import TreeStats
from src.games.state.State import State
import time

from src.games.mcts.MctsNode import MctsNode


class MctsRun:
    def __init__(self) -> None:
        # time
        self.select_time = 0
        self.expand_time = 0
        self.simulate_time = 0
        self.backpropagation_time = 0
        self.run_time = 0

    def __repr__(self):
        return 'time:\n select={}\n expand={}\n simulate={}\n backpropagation={}\n total={}'.format(
            self.select_time,
            self.expand_time,
            self.simulate_time,
            self.backpropagation_time,
            self.run_time
        )


class Mcts:
    def __init__(self, initial_state: State = None, root_node: MctsNode = None) -> None:
        if root_node is None:
            root_node = MctsNode(initial_state)
        self.tree = TreeStats(root_node)
        self.stats = MctsRun()

    def run(self, n):
        start = time.time()
        for i in range(0, n):
            selected = self.select()
            expansion = self.expand(selected)
            reward = self.simulate(expansion)
            self.backpropagation(expansion, reward)
        delta = time.time() - start
        self.stats.run_time += delta
        return delta

    def select(self) -> MctsNode:
        start = time.time()

        def side_exploitation_score(node: MctsNode):
            side = -1 if node.state.board.current_turn % 2 == 0 else 1
            return side * node.exploitation_score()

        def node_score(node: MctsNode):
            return side_exploitation_score(node) + node.exploration_score()

        current = self.tree.root
        while current.games > 0 and not current.state.is_terminal:
            candidates = current.children().values()

            next_current = max(candidates, key=side_exploitation_score)
            if next_current.state.is_terminal:
                break

            next_current = max(candidates, key=node_score)
            if next_current == current:
                break
            current = next_current
        self.stats.select_time += time.time() - start
        return current

    def expand(self, node: MctsNode):
        start = time.time()
        if node.state.is_terminal:
            expansion = self.tree.register_node(node)
        elif node.games == 0:
            expansion = self.tree.register_node(node)
        else:
            children = node.children()
            child = random.choice(list(children.values()))
            expansion = self.tree.register_node(child)
        self.stats.expand_time += time.time() - start
        return expansion

    def simulate(self, node: MctsNode):
        start = time.time()
        if node.state.is_terminal:
            self.stats.simulate_time += time.time() - start
            return node.state.terminal_result
        current = node.state.copy()
        while not current.is_terminal:
            action = random.choice(current.actions())
            current.apply(action)
        self.stats.simulate_time += time.time() - start
        return current.terminal_result

    def backpropagation(self, node: MctsNode, score):
        start = time.time()
        depth = 0
        while node is not self.tree.root:
            node.score += score
            node.games += 1
            node = node.parent
            depth += 1
        node.score += score
        node.games += 1
        self.stats.backpropagation_time += time.time() - start

    def __repr__(self):
        return "{0}\n{1}\n{2}\n".format(
            str(self.tree),
            str(self.thinking_stats()),
            str(self.stats))

    def thinking_stats(self):
        return pprint.pformat(self.tree.root.children())
