from src.games.Game import Game
from src.games.Policy import Human, MctsPolicy, MctsPolicy2
from src.games.state.State import State
from src.power4.P4Rules import P4Rules

human = Human('human')
computer = MctsPolicy2(n=2000, after_play=lambda mcts: print(mcts))


def show_state(state: State):
    print(' [0 1 2 3 4 5 6]')
    print(state)
    print(' [0 1 2 3 4 5 6]')


def power4_play():
    player1 = computer
    player2 = computer

    game = Game(P4Rules.initial_state(), player1, player2, before_play=show_state)
    game.run()

    print(game.state)
    print('result : {}'.format(game.state.terminal_result))


# cProfile.run('test_mcts()')

power4_play()
