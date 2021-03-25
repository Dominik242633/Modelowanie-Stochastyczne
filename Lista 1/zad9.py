import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lng


rho = [0, 0.25, 0.5, 0.75, 0.95]
Z = np.random.normal(0, 1, (2, 10 ** 5))
for i in rho:
    gamma = [[1, i],
             [i, 1]]

    C = lng.cholesky(gamma)
    M = np.dot(C.T, Z)

    plt.scatter(M[0, :], M[1, :])
    plt.title('rho = ' + str(i))
    plt.xlabel('x', fontdict={'size': 16})
    plt.ylabel('y', fontdict={'size': 16})
    plt.show()
