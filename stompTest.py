# test of stagewise orthogonal matching persuit (Donoho et al.)
import numpy as np

k = 5
N = 32

# create a k-spare N dimensional vector
x = np.random.rand(N)
random_indicies = np.random.choice(N, k, replace=False)
for i in range(N):
    if not i in random_indicies:
        x[i] = 0

print(x)
