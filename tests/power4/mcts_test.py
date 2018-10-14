from src.games.Policy import Human, MctsPolicy
from src.games.util.Mcts import Mcts
from src.power4.P4Rules import P4Rules

playouts = 500
print('Computer will simulate {} games before playing !'.format(playouts))

state = P4Rules.initial_state()

mcts = Mcts(initial_state=state)

while not state.is_terminal:
    print(state)
    time_spent = mcts.run(playouts - mcts.root_node.games)
    print('time spent: {}'.format(time_spent))
    mcts.stats()
    action = mcts.root_node.best_action()
    state.apply(action)

    next_node = mcts.root_node.children()[action]
    mcts = Mcts(root_node=next_node)
