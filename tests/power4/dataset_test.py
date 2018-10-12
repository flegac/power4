from src.dataset.dataset import Dataset

X_array = [
    [0, 1, 55, 3, 4],
    [4, 3, 2, 1, 0]
]

Y_array = [
    [1, 2, 3],
    [4, 5, 6]
]

db = Dataset()

db.write(X_array, Y_array)

data = db.read()

print(data)
