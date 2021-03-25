import numpy as np
import matplotlib.pyplot as plt


mu = 0.04
sigma = 0.4
dt = 1

quantity = 1000
X = np.arange(0, quantity)

Y = np.zeros((quantity, 1))
Y[0] = 10

Y1 = np.zeros((quantity, 1))
Y1[0] = 10

Y2 = np.zeros((quantity, 1))
Y2[0] = 10

Y3 = np.zeros((quantity, 1))
Y3[0] = 10

for i in range(1, quantity):
    Y[i] = Y[i-1] + mu * dt
    Y1[i] = Y1[i-1] + np.random.normal(mu * dt, sigma ** 2 * dt)
    Y2[i] = Y2[i-1] + np.random.normal(mu * dt, sigma ** 2 * dt)
    Y3[i] = Y3[i-1] + np.random.normal(mu * dt, sigma ** 2 * dt)

plt.plot(X, Y, label='Trend')
plt.plot(X, Y1, label='Trajektoria 1')
plt.plot(X, Y2, label='Trajektoria 2')
plt.plot(X, Y3, label='Trajektoria 3')
plt.legend(loc='lower right', frameon=False)
plt.title('Arytmetyczny ruch Browna')
plt.xlabel('x', fontdict={'size': 16})
plt.ylabel('F(x)', fontdict={'size': 16})
plt.show()
