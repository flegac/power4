from src.deep.pipeline.dataset import Dataset
import numpy as np

N = 1000
B = 3
W = 19
H = 19

x_shape = [N, B, H, W]
X = np.random.random(x_shape).astype(np.float32)

y_shape = [N, W]
Y = np.random.random(y_shape).astype(np.float32)

db = Dataset()
db.write(X, Y)

dataset = db.read()

for x in dataset:
    print(x[0])
    print(x[1])
