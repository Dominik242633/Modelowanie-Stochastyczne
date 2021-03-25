import numpy as np
import matplotlib.pyplot as plt


alpha = 0.1
beta = 8
sigma = 0.4
dt = 1

quantity = 200
X = np.arange(0, quantity)

Y = np.zeros((quantity, 1))
Y[0] = 20

Y1 = np.zeros((quantity, 1))
Y1[0] = 20

Y2 = np.zeros((quantity, 1))
Y2[0] = 20

Y3 = np.zeros((quantity, 1))
Y3[0] = 20

for i in range(1, quantity):
    Y[i] = Y[i-1] + alpha * (beta - Y[i-1]) * dt
    Y1[i] = Y1[i-1] + alpha * (beta - Y1[i-1]) * dt + np.random.normal(0, sigma ** 2 * dt)
    Y2[i] = Y2[i-1] + alpha * (beta - Y2[i-1]) * dt + np.random.normal(0, sigma ** 2 * dt)
    Y3[i] = Y3[i-1] + alpha * (beta - Y3[i-1]) * dt + np.random.normal(0, sigma ** 2 * dt)

plt.plot(X, Y, label='Trend')
plt.plot(X, Y1, label='Trajektoria 1')
plt.plot(X, Y2, label='Trajektoria 2')
plt.plot(X, Y3, label='Trajektoria 3')
plt.legend(loc='upper right', frameon=False)
plt.title('Dyfuzja powracająca do średniej')
plt.xlabel('t', fontdict={'size': 16})
plt.ylabel('F(t)', fontdict={'size': 16})
plt.show()
