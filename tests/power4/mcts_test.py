from src.games.Game import Game
from src.games.Policy import Human, MctsPolicy
from src.games.state.State import State
from src.power4.P4Rules import P4Rules

playouts = 500
print('Computer will simulate {} games before playing !'.format(playouts))

human = Human('human')
computer = MctsPolicy(n=playouts)


def show_state(state: State):
    print(' [0 1 2 3 4 5 6]')
    print(state)
    print(' [0 1 2 3 4 5 6]')


def power4_play():
    player1 = human
    player2 = computer

    game = Game(P4Rules.start(), player1, player2, turn_callback=show_state)
    game.run()

    print(game.state)
    print('result : {}'.format(game.state.terminal_result))


# cProfile.run('test_mcts()')

power4_play()
