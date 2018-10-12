import os

from src.dataset.dataset import Dataset

current_path = os.getcwd()
root_path = os.path.join(current_path, 'dataset', 'dataset.txt')

print('root_path: ', os.path.dirname(root_path))

db = Dataset(root_path, x_size=5, y_size=3)

X = list(range(0, 5))
Y = list(range(0, 3))

db.save(X, Y)
db.save(X, Y)
db.save(X, Y)
db.save(X, Y)
db.save(X, Y)


x, y = db.load()
print('x=', x)
print('y=', y)
