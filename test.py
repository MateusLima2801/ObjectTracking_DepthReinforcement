import numpy as np
import scipy.optimize as opt

# Matrix A
A = np.array([
    [1, 10, 20],
    [7, 10, -1],
    [8, 4, 10],
    [2, 15, 10]
])

# Matrix B
B = np.array([
    [5, 10, 20, 1],
    [4, 8, -1, 7],
    [8, 2, 10, 5]
])

res_A = opt.linear_sum_assignment(A)
res_B = opt.linear_sum_assignment(B)

print(res_A)
print(res_B)