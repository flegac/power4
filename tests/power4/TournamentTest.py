from functools import reduce
from src.games.Player import Player
import numpy as np
import matplotlib.pyplot as plt

from src.power4.Power4 import EvaluatorP4, P4Rules
from src.games.Tournament import Tournament

player = Player(EvaluatorP4(), depth=3, exploration=0.33)
players = [player] * 2

state = P4Rules.start()

tournament = Tournament(state, game_per_player=20)
tournament.set_players(*players)
tournament.run()

print('done')

for game in tournament.games:
    print(len(game), ' ', game)

game_lengths = [len(x) for x in tournament.games]
l_min = reduce(lambda x, y: min(x, y), game_lengths)
l_max = reduce(lambda x, y: max(x, y), game_lengths)
avg = reduce(lambda x, y: x + y, game_lengths) / len(game_lengths)

print('min length: ', l_min)
print('max length: ', l_max)
print('avg length: ', avg)
print('total games: ', len(game_lengths))
print('total _positions: ', avg * len(game_lengths))

mu, sigma = 200, 25
x = mu + sigma * np.random.randn(10000)
n, bins, patches = plt.hist(game_lengths)
plt.show()
