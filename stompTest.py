# test of stagewise orthogonal matching persuit (Donoho et al.)
import numpy as np

# parameters of problem: k is sparsity and N is total dimension
k = 2
N = 8
print('k = {}, N = {}\n'.format(k, N))

# create a k-spare N dimensional vector
x = np.random.rand(N)
random_indicies = np.random.choice(N, k, replace=False)
for index in range(N):
    if not index in random_indicies:
        x[index] = 0
print('x = '+str(x)+'\n')

# create random Uniform Spherical Ensemble measurement matrix (cols are random
# points on unit sphere)
phi = np.random.rand(N,N)
for col_index in range(N):
    # normalise each column (i.e. make sure each column is point on unit sphere)
    col_norm = np.linalg.norm(phi[:,col_index])
    phi[:,col_index] /= col_norm
print('phi = '+str(phi)+'\n')

# calculate y (vector of measurements)
y = np.matmul(phi, x)
print('y = '+str(y) +'\n')
