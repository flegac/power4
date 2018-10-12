import itertools

from src.games.Game import Game
from src.games.state.State import State
from src.games.Policy import Policy


class Tournament:
    def __init__(self, state: State, game_per_player: int) -> None:
        self.state = state
        self.games = []
        self.game_per_player = game_per_player
        self.players = []

    def set_players(self, *players: [Policy]):
        self.players = players
        for player in players:
            player.score = 0
            player.games = 0

    def run(self):
        for pair in itertools.combinations(self.players, 2):
            p1, p2 = pair
            for i in range(self.game_per_player):
                game = Game(self.state.copy(), p1, p2)
                game.run()
                p1.score += game.state.terminal_result
                p1.games += 1
                p2.score -= game.state.terminal_result
                p2.games += 1

                self.games.append(game.history)
                print(game.state)
                print('{} vs {} : {}'.format(p1.name(), p2.name(), game.state.terminal_result))
                print()
                p1, p2 = p2, p1
