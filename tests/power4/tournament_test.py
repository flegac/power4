from src.games.Policy import MctsPolicy
from src.games.Tournament import Tournament
from src.power4.P4Rules import P4Rules

evaluator = None

tournament = Tournament(P4Rules.initial_state())

players = [MctsPolicy(n=100 + 1000 * i) for i in range(0, 2)]

tournament \
    .with_players(*players) \
    .run(game_per_player=5) \
    .stats()
