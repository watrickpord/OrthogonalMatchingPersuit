# test of stagewise orthogonal matching persuit (Donoho et al.)
import numpy as np
import math

# parameters of problem: k is sparsity and N is total dimension, t is threshold multiplier
k = 4
N = 32
t = 1.1
print('k = {}, N = {}, t = {}\n'.format(k, N, t))

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
r0 = np.matmul(np.transpose(phi), y0)
print('r0 = '+str(r0)+'\n')

# calculate formal noise level and threshold for residual
sigma1 = np.linalg.norm(r0)/math.sqrt(N)
print('sigma1 = '+str(sigma1))
tsig1 = sigma1*t
print('t*sigma1 = '+str(tsig1)+'\n')

# take residual elements greater than the threshold, and note their indicies
j1 = r0[r0 > tsig1]
print('j1 = '+str(j1))
i1 = np.arange(N)[r0 > tsig1]
print('i1 = '+str(i1)+'\n')

# matrix phi_i1 is the columns in i1 or phi
phi_i1 = np.array([phi[:,col_index] for col_index in i1]).transpose()
print('phi_i1 = '+str(phi_i1)+'\n')

# project y onto columns of phi belonging to the support I (phi_i1), giving new x estimate
x1 = np.matmul(np.linalg.inv(np.matmul(phi_i1.transpose(), phi_i1)), phi_i1.transpose(), y)
print('x1 = '+str(x1)+'\n')

# update residuals: r' = y - phi x_n
r1 = y - np.matmul(phi, x1)
print('r1 = '+str(r1)+'\n')
