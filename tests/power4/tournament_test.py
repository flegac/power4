from src.games.Policy import MctsPolicy
from src.games.Tournament import Tournament
from src.power4.P4Rules import P4Rules

evaluator = None

players = [MctsPolicy(n=100 + 1000 * i) for i in range(0, 2)]

state = P4Rules.start()

tournament = Tournament(state, game_per_player=5)
tournament.set_players(*players)
tournament.run()

tournament.stats()
