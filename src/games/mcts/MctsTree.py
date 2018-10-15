import pprint

from src.games.mcts.MctsNode import MctsNode


class MctsTree:
    def __init__(self, root: MctsNode) -> None:
        self.root = root
        self.tree = {root}
        self.depth = 0

    def register_node(self, node: MctsNode):
        self.tree.add(node)
        self.depth = max(self.depth, node.depth)
        return node

    def __repr__(self):
        return pprint.pformat({
            'tree final size': len(self.tree),
            'tree depth': self.depth,
            'tree root playouts': self.root.games
        }, indent=2)
