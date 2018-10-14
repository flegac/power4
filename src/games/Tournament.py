import itertools
from functools import reduce

from src.games.Game import Game
from src.games.state.State import State
from src.games.Policy import Policy

import matplotlib.pyplot as plt


def _do_nothing(game: Game):
    pass


def _show_game(game: Game):
    print(str(game) + '\n')


class Tournament:
    def __init__(self, state: State, game_per_player: int,
                 end_game_callback=_show_game) -> None:
        self.state = state
        self.game_per_player = game_per_player
        self.players = []
        self.end_game_callback = end_game_callback
        self.game_lengths = {}

    def set_players(self, *players: [Policy]):
        self.players = players
        for player in players:
            player.victories = 0
            player.defeats = 0
            player.games = 0
            player.score = 0

    def run(self):
        for pair in itertools.combinations(self.players, 2):
            p1, p2 = pair
            for i in range(2 * self.game_per_player):
                game = Game(self.state.copy(), p1, p2)
                game.run()

                self.update_scores(game)

                length = len(game.history)
                if self.game_lengths.get(length) is None:
                    self.game_lengths[length] = 0
                self.game_lengths[length] += 1

                self.end_game_callback(game)
                p1, p2 = p2, p1

    def update_scores(self, game):
        p1, p2 = game.policies

        if game.state.terminal_result > 0:
            p1.victories += 1
            p2.defeats += 1
        elif game.state.terminal_result < 0:
            p1.defeats += 1
            p2.victories += 1
        p1.score += game.state.terminal_result
        p1.games += 1
        p2.score -= game.state.terminal_result
        p2.games += 1

    def stats(self):
        game_lengths = self.game_lengths
        total_games = len(game_lengths)
        visted_positions = sum(game_lengths)
        l_min = reduce(lambda x, y: min(x, y), game_lengths)
        l_max = reduce(lambda x, y: max(x, y), game_lengths)
        average = visted_positions / total_games

        print('min length:        ', l_min)
        print('max length:        ', l_max)
        print('average length:    ', average)
        print('total games:       ', total_games)
        print('visited positions: ', visted_positions)

        for player in sorted(self.players, key=lambda x: x.score):
            print('{p} [ v:{p.victories} d:{p.defeats} | {p.score}/{p.games} ] '.format(p=player))

        plt.bar(range(len(game_lengths)), list(game_lengths.values()), align='center')
        plt.xticks(range(len(game_lengths)), list(game_lengths.keys()))
        plt.show()
