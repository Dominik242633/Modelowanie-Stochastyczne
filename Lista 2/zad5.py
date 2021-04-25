import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def init_parameters(N, T, I_0):
    sS = np.zeros((T, 1))
    sE = np.zeros((T, 1))
    sI = np.zeros((T, 1))
    sR = np.zeros((T, 1))

    sI[0] = I_0
    sS[0] = N - I_0

    return [sS, sE, sI, sR]

def set_plot_properties(title):
    plt.title(title)
    plt.legend(loc='best', frameon=False)
    plt.xlabel('Czas', fontdict={'size': 16})
    plt.ylabel('Liczba przypadków', fontdict={'size': 16})
    plt.show()


def plot_model(y, title, N):
    labels = ['$S_t$', '$E_t$', '$I_t$', '$R_t$']

    for i in range(len(y)):
        if np.mean(y[i]) != 0:
            plt.plot(np.arange(0, len(y[i]) / N, 1 / N), y[i], label=labels[i])
    set_plot_properties(title + '\nSkala liniowa')

    for i in range(len(y)):
        if np.mean(y[i]) != 0:
            plt.plot(np.arange(0, len(y[i]) / N, 1 / N), y[i], label=labels[i])
    plt.xscale("log")
    plt.yscale("log")
    set_plot_properties(title + '\nSkala logarytmiczna')


def stochastic_SIR(parameters, beta, gamma, eta, sigma):
    sS, sE, sI, sR = parameters
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
            sI[i + 1] = sI[i + 1] - 1
            sR[i + 1] = sR[i] + 1
        else:
            sR[i + 1] = sR[i]

    return [sS, sE, sI, sR]


def stochastic_SI(parameters, beta, gamma, eta, sigma):
    sS, sE, sI, sR = parameters
    u1 = np.random.uniform(0, 1, (T, 1))

    for i in range(T - 1):
        if u1[i] < beta * sI[i] * sS[i] / N ** 2:
            sS[i + 1] = sS[i] - 1
            sI[i + 1] = sI[i] + 1
        else:
            sS[i + 1] = sS[i]
            sI[i + 1] = sI[i]

    return [sS, sE, sI, sR]


def stochastic_SIS(parameters, beta, gamma, eta, sigma):
    sS, sE, sI, sR = parameters
    u1 = np.random.uniform(0, 1, (T, 1))
    u2 = np.random.uniform(0, 1, (T, 1))

    for i in range(T - 1):
        # S -> I
        if u1[i] < ((beta * sI[i] * sS[i]) / N ** 2):
            sS[i + 1] = sS[i] - 1
            sI[i + 1] = sI[i] + 1
        else:
            sS[i + 1] = sS[i]
            sI[i + 1] = sI[i]
        # I -> S
        if u2[i] < ((gamma * sI[i]) / N):
            sI[i + 1] = sI[i + 1] - 1
            sS[i + 1] = sS[i + 1] + 1

    return [sS, sE, sI, sR]


def stochastic_SIRS(parameters, beta, gamma, eta, sigma):
    sS, sE, sI, sR = parameters
    u1 = np.random.uniform(0, 1, (T, 1))
    u2 = np.random.uniform(0, 1, (T, 1))
    u3 = np.random.uniform(0, 1, (T, 1))

    for i in range(T - 1):
        # S -> I
        if u1[i] < (beta * sI[i] * sS[i] / N ** 2):
            sS[i + 1] = sS[i] - 1
            sI[i + 1] = sI[i] + 1
        else:
            sS[i + 1] = sS[i]
            sI[i + 1] = sI[i]
        # I -> R
        if u2[i] < (gamma * sI[i] / N):
            sI[i + 1] = sI[i + 1] - 1
            sR[i + 1] = sR[i] + 1
        else:
            sR[i + 1] = sR[i]
        # R -> S
        if u3[i] < (eta * sR[i] / N):
            sR[i + 1] = sR[i + 1] - 1
            sS[i + 1] = sS[i + 1] + 1

    return [sS, sE, sI, sR]


def stochastic_SEIR(parameters, beta, gamma, eta, sigma):
    sS, sE, sI, sR = parameters
    u1 = np.random.uniform(0, 1, (T, 1))
    u2 = np.random.uniform(0, 1, (T, 1))
    u3 = np.random.uniform(0, 1, (T, 1))

    for i in range(T - 1):
        # S -> E
        if u1[i] < (beta * sI[i] * sS[i] / N ** 2):
            sS[i + 1] = sS[i] - 1
            sE[i + 1] = sE[i] + 1
        else:
            sS[i + 1] = sS[i]
            sE[i + 1] = sE[i]
        # E -> I
        if u2[i] < (sigma * sE[i] / N):
            sE[i + 1] = sE[i + 1] - 1
            sI[i + 1] = sI[i] + 1
        else:
            sI[i + 1] = sI[i]
        # I -> R
        if u3[i] < (gamma * sI[i] / N):
            sI[i + 1] = sI[i + 1] - 1
            sR[i + 1] = sR[i] + 1
        else:
            sR[i + 1] = sR[i]

    return [sS, sE, sI, sR]


def stochastic_SEIRS(parameters, beta, gamma, eta, sigma):
    sS, sE, sI, sR = parameters
    u1 = np.random.uniform(0, 1, (T, 1))
    u2 = np.random.uniform(0, 1, (T, 1))
    u3 = np.random.uniform(0, 1, (T, 1))
    u4 = np.random.uniform(0, 1, (T, 1))

    for i in range(T - 1):
        # S -> E
        if u1[i] < (beta * sI[i] * sS[i] / N ** 2):
            sS[i + 1] = sS[i] - 1
            sE[i + 1] = sE[i] + 1
        else:
            sS[i + 1] = sS[i]
            sE[i + 1] = sE[i]
        # E -> I
        if u2[i] < (sigma * sE[i] / N):
            sE[i + 1] = sE[i + 1] - 1
            sI[i + 1] = sI[i] + 1
        else:
            sI[i + 1] = sI[i]
        # I -> R
        if u3[i] < (gamma * sI[i] / N):
            sI[i + 1] = sI[i + 1] - 1
            sR[i + 1] = sR[i] + 1
        else:
            sR[i + 1] = sR[i]
        # R -> S
        if u4[i] < (eta * sR[i] / N):
            sR[i + 1] = sR[i + 1] - 1
            sS[i + 1] = sS[i + 1] + 1

    return [sS, sE, sI, sR]


beta = 0.5
gamma = 0.1
eta = gamma
sigma = gamma
N = 1000
T = 100 * N
I_0 = 1

if __name__ == '__main__':
    plot_model(stochastic_SIR(init_parameters(N, T, I_0), beta, gamma, eta, sigma), "Stochastyczny SIR", N)

    plot_model(stochastic_SI(init_parameters(N, T, I_0), beta, gamma, eta, sigma), "Stochastyczny SI", N)

    plot_model(stochastic_SIS(init_parameters(N, T, I_0), beta, gamma, eta, sigma), "Stochastyczny SIS", N)

    plot_model(stochastic_SIRS(init_parameters(N, T, I_0), beta, gamma, eta, sigma), "Stochastyczny SIRS", N)

    plot_model(stochastic_SEIR(init_parameters(N, T, I_0), beta, gamma, eta, sigma), "Stochastyczny SEIR", N)

    plot_model(stochastic_SEIRS(init_parameters(N, T, I_0), beta, gamma, eta, sigma), "Stochastyczny SEIRS", N)
