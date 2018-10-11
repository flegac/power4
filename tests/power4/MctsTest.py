from src.games.util.mcts import MCTS
from src.power4.Power4 import P4Rules


def power4_play(player_side):
    level = int(input('choose level : [1-10]'))
    level = max(level, 1)
    level = min(level, 100)
    playouts = 100 * level

    print('Computer will simulate {} games before playing !'.format(playouts))

    state = P4Rules.start()
    while not state.is_terminal:
        mcts = MCTS(state=state)
        print(' [0 1 2 3 4 5 6]')
        print(state)
        print(' [0 1 2 3 4 5 6]')
        current_side = state.board.current_turn % 2
        if current_side == player_side:
            actions = state.actions()
            player_action = None
            while player_action not in actions:
                try:
                    player_action = int(input('Your turn : choose from {}'.format(actions)))
                except:
                    print('invalid move !')
            state.next(player_action)
        else:
            print('thinking ...')
            mcts.run(n=playouts, m=10)
            print('tree final size: {}'.format(len(mcts.tree)))
            print('games : ', mcts.games)
            print('scores :')
            for node in mcts.root.children():
                print('{}'.format(node))
            template = 'time:\n select={}\n expand={}\n simulate={}\n backpropagation={}\n'
            print(template.format(
                mcts.select_time,
                mcts.expand_time,
                mcts.simulate_time,
                mcts.backpropagation_time))
            child = mcts.root.best_move()
            state = child.state
    print(state)
    print('result : {}'.format(state.terminal_result))
    if state.terminal_result == 0:
        print('You draw with the computer ... Awesome !')
    elif state.terminal_result == 1:
        print('Good : you beat the computer !')
    else:
        print('Hahaha ! The COMPUTER BEAT YOU !! It was level {}/10 ...'.format(level))


# cProfile.run('test_mcts()')

try:
    power4_play(player_side=None)
except Exception as e:
    print(e)
