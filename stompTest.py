# test of stagewise orthogonal matching persuit (Donoho et al.)
import numpy as np

# parameters of problem: k is sparsity and N is total dimension
k = 4
N = 32
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
print('y = '+str(y)+'\n')

# set up initial values for problem: x0 = 0, y0 = y
x0 = np.zeros(N)
y0 = y.copy()
print('x0 = '+str(x0))
print('y0 = '+str(y0)+'\n')

# first redidual vector, r1 = phi^T y
r1 = np.matmul(np.transpose(phi), y0)
print('r1 = '+str(r1)+'\n')
