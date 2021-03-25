import numpy as np
import matplotlib.pyplot as plt


def set_plot_properties(x, y, title, legend_loc, x_label, y_label):
    plt.plot(x, y[:, 0], label='Trajektoria 1')
    plt.plot(x, y[:, 1], label='Trajektoria 2')
    plt.plot(x, y[:, 2], label='Trajektoria 3')
    plt.plot(x, y[:, 3], label='Trend')
    plt.title(title)
    plt.legend(loc=legend_loc, frameon=False)
    plt.xlabel(x_label, fontdict={'size': 16})
    plt.ylabel(y_label, fontdict={'size': 16})
    plt.show()


dt = 1
x = np.loadtxt("DJIA.txt", comments="#", delimiter="\t", unpack=False)
y = x[:, 1]
x = np.arange(0, len(y), 1)

roznica_szeregu = np.zeros((len(y), 1))
zwrot = np.zeros((len(y), 1))

for i in range(1, len(y)):
    roznica_szeregu[i] = y[i] - y[i - 1]
    zwrot[i] = (y[i] - y[i - 1]) / y[i - 1]

mu_ABM = np.mean(roznica_szeregu)
sigma_ABM = np.std(roznica_szeregu)

mu_GBM = np.mean(zwrot)
sigma_GBM = np.std(zwrot)

ABM = np.zeros((len(y), 4))
GBM = np.zeros((len(y), 4))

ABM[0, :] = 51.6
GBM[0, :] = 51.6

for i in range(1, len(y)):
    ABM[i, 0] = ABM[i-1, 0] + (mu_ABM * dt + sigma_ABM * np.random.normal(0, 1))
    ABM[i, 1] = ABM[i - 1, 1] + (mu_ABM * dt + sigma_ABM * np.random.normal(0, 1))
    ABM[i, 2] = ABM[i - 1, 2] + (mu_ABM * dt + sigma_ABM * np.random.normal(0, 1))
    ABM[i, 3] = ABM[i - 1, 3] + (mu_ABM * dt)

    GBM[i, 0] = GBM[i-1, 0] * np.exp((mu_GBM - (sigma_GBM ** 2) / 2) + sigma_GBM * np.random.normal(0, 1))
    GBM[i, 1] = GBM[i-1, 1] * np.exp((mu_GBM - (sigma_GBM ** 2) / 2) + sigma_GBM * np.random.normal(0, 1))
    GBM[i, 2] = GBM[i-1, 2] * np.exp((mu_GBM - (sigma_GBM ** 2) / 2) + sigma_GBM * np.random.normal(0, 1))
    GBM[i, 3] = GBM[i-1, 3] * np.exp((mu_GBM - (sigma_GBM ** 2) / 2))

plt.plot(x, ABM[:, 0], label='ABM')
plt.plot(x, GBM[:, 0], label='GBM')
plt.plot(x, y, label='DJIA')
plt.legend(loc='upper left', frameon=False)
plt.title('Porównanie ABM, GBM i faktycznego indeksu DJIA')
plt.xlabel('t', fontdict={'size': 16})
plt.ylabel('F(t)', fontdict={'size': 16})
plt.show()

set_plot_properties(x, ABM, 'Porównanie wygenerowanych trajektorii ABM',
                    'upper left', 't', 'F(t)')

set_plot_properties(x, GBM, 'Porównanie wygenerowanych trajektorii GBM',
                    'upper left', 't', 'F(t)')

