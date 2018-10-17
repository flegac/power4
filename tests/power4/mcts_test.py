from src.games.mcts.Mcts import Mcts
from src.power4.P4Rules import P4Rules

playouts = 2000
print('Computer will simulate {} games before playing !'.format(playouts))

state = P4Rules.initial_state()

mcts = Mcts(initial_state=state)

while not state.is_terminal:
    print(state)
    mcts.run(playouts)
    print('time spent: {}'.format(mcts.stats.run_time))
    print(mcts)
    action = mcts.tree.root.best_action()
    state.apply(action)

    next_node = mcts.tree.root.children()[action]
    mcts = Mcts(root_node=next_node)
