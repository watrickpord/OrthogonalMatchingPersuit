# test of orthogonal matching persuit algorithm
# as per stagewise OMP, but simply add largest component at each step
import numpy as np
import math

# parameters of problem: k is sparsity and N is total dimension, t is threshold multiplier
k = 2
N = 16
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



# ----------------------- OMP algorithm ------------------------
for n in range(k):
    # Step 1: initialise residuals with r_0 = y and colums set C = null
    R = [y]
    C = []

    # Step 2: find the column what maximises projection onto residuals (y in first instance)
    max_projection = 0
    max_index = -1
    for i in range(N):
        # take dot product of ith colum and residual
        projection_i = np.dot(phi[:, i], R[-1])
        print('proj[{}] = {}'.format(i, projection_i))
        # compare projection against ith column to current max, if bigger save it
        max_index = i if (projection_i > max_projection) else max_index
        max_projection = projection_i if (max_index == i) else max_projection
    print('max_projection = {}, index = {}'.format(max_projection, max_index))

    # update list of column indicies with max_index
    C.append(max_index)
    print(C)

    # step 4: project residual onto columns of index C and subtract this new
    # estimate from the residual to get next step residual
    phi_cols = np.array([phi[:,col_index] for col_index in C]).transpose()
    print('\nNonzero Phi Cols')
    print(phi_cols)
    P_i = np.matmul(phi_cols, np.matmul(np.linalg.inv(np.matmul(phi_cols.transpose(), phi_cols)), phi_cols.transpose()))

    print('\nPi = ')
    print(P_i)
    # update residuals r' = y - P_i y
    R.append(np.matmul(np.identity(N) - P_i, y))
    print('\nNew residual = {}'.format(R[-1]))
