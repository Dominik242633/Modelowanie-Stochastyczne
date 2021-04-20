import numpy as np
import matplotlib.pyplot as plt


def init_parameters(I0, N):
    Y = np.zeros((4, T))
    Y[2, 0] = I0
    Y[0, 0] = N - Y[2, 0]

    return Y


def set_plot_properties(title):
    plt.title(title)
    plt.legend(loc='best', frameon=False)
    plt.xlabel('Czas', fontdict={'size': 16})
    plt.ylabel('Liczba przypadków', fontdict={'size': 16})
    plt.show()


def plot_model(y, title):
    labels = ['$S_t$', '$E_t$', '$I_t$', '$R_t$']

    for i in range(len(y)):
        if np.mean(y[i]) != 0:
            plt.plot(np.arange(0, len(y[i])), y[i], label=labels[i])
    set_plot_properties(title + '\nSkala liniowa')

    for i in range(len(y)):
        if np.mean(y[i]) != 0:
            plt.plot(np.arange(0, len(y[i])), y[i], label=labels[i])
    plt.xscale("log")
    plt.yscale("log")
    set_plot_properties(title + '\nSkala logarytmiczna')


beta = 0.5
gamma = 0.1
eta = gamma
sigma = gamma
N = 1000
T = 100 * N
dt = 1
I_0 = 1

sS = np.zeros((T, 1))
sE = np.zeros((T, 1))
sI = np.zeros((T, 1))
sR = np.zeros((T, 1))

sI[0] = I_0
sS[0] = N - I_0

u1 = np.random.uniform(0, 1, (T, 1))
u2 = np.random.uniform(0, 1, (T, 1))

for i in range(T - 1):
    if u1[i] < beta * sI[i] * sS[i] / N ** 2:
        sS[i + 1] = sS[i] - 1
        sI[i + 1] = sI[i] + 1
    else:
        sS[i + 1] = sS[i]
        sI[i + 1] = sI[i]

    if u2[i] < gamma * sI[i] / N:
        sI[i + 1] = sI[i+1] - 1
        sR[i + 1] = sR[i] + 1
    else:
        sR[i + 1] = sR[i]

plot_model([sS, sE, sI, sR], "Stochastyczny SIR")
