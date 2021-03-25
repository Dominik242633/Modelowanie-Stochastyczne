import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lng


rho = [0, 0.25, 0.5, 0.75, 0.95]
dt = 1
mu = 0.04
sigma = 0.4
quantity = 1000

x = np.arange(0, quantity, 1)
y = np.zeros((quantity, 2))
y[0, :] = 10
Z = np.random.normal(0, 1, (2, quantity))

for i in rho:
    gamma = [[1, i],
             [i, 1]]

    C = lng.cholesky(gamma)
    M = np.dot(C.T, Z)

    for j in range(1, quantity):
        y[j, :] = y[j - 1, :] + (mu * dt + sigma * M[:, j - 1])

    plt.plot(x, y[:, 0], label='Trajektoria 1')
    plt.plot(x, y[:, 1], label='Trajektoria 2')
    plt.title('rho = ' + str(i))
    plt.legend(loc='lower right', frameon=False)
    plt.xlabel('t', fontdict={'size': 16})
    plt.ylabel('F(t)', fontdict={'size': 16})
    plt.show()
