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


def SIR(N, T, dt, gamma, beta, I0):
    Y = init_parameters(I0, N)

    for i in range(1, T):
        Y[0, i] = Y[0, i - 1] + dt * (-(beta / N) * Y[2, i - 1] * Y[0, i - 1])
        Y[2, i] = Y[2, i - 1] + dt * ((beta / N) * Y[2, i - 1] * Y[0, i - 1] - gamma * Y[2, i - 1])
        Y[3, i] = Y[3, i - 1] + dt * (gamma * Y[2, i - 1])

    return Y


def SI(N, T, dt, beta, I0):
    Y = init_parameters(I0, N)

    for i in range(1, T):
        Y[0, i] = Y[0, i - 1] + dt * (-(beta / N) * Y[2, i - 1] * Y[0, i - 1])
        Y[2, i] = Y[2, i - 1] + dt * ((beta / N) * Y[2, i - 1] * Y[0, i - 1])

    return Y


def SIS(N, T, dt, gamma, beta, I0):
    Y = init_parameters(I0, N)

    for i in range(1, T):
        Y[0, i] = Y[0, i - 1] + dt * (-(beta / N) * Y[2, i - 1] * Y[0, i - 1] + gamma * Y[2, i - 1])
        Y[2, i] = Y[2, i - 1] + dt * ((beta / N) * Y[2, i - 1] * Y[0, i - 1] - gamma * Y[2, i - 1])

    return Y


def SIRS(N, T, dt, gamma, beta, eta, I0):
    Y = init_parameters(I0, N)

    for i in range(1, T):
        Y[0, i] = Y[0, i - 1] + dt * (-(beta / N) * Y[2, i - 1] * Y[0, i - 1] + eta * Y[3, i - 1])
        Y[2, i] = Y[2, i - 1] + dt * ((beta / N) * Y[2, i - 1] * Y[0, i - 1] - gamma * Y[2, i - 1])
        Y[3, i] = Y[3, i - 1] + dt * (gamma * Y[2, i - 1] - eta * Y[3, i - 1])

    return Y


def SEIR(N, T, dt, gamma, beta, sigma, I0):
    Y = init_parameters(I0, N)

    for i in range(1, T):
        Y[0, i] = Y[0, i - 1] + dt * (-(beta / N) * Y[2, i - 1] * Y[0, i - 1])
        Y[1, i] = Y[1, i - 1] + dt * ((beta / N) * Y[2, i - 1] * Y[0, i - 1] - sigma * Y[1, i - 1])
        Y[2, i] = Y[2, i - 1] + dt * (sigma * Y[1, i - 1] - gamma * Y[2, i - 1])
        Y[3, i] = Y[3, i - 1] + dt * (gamma * Y[2, i - 1])

    return Y


def SEIRS(N, T, dt, gamma, beta, sigma, I0):
    Y = init_parameters(I0, N)

    for i in range(1, T):
        Y[0, i] = Y[0, i - 1] + dt * (-(beta / N) * Y[2, i - 1] * Y[0, i - 1] + eta * Y[3, i - 1])
        Y[1, i] = Y[1, i - 1] + dt * ((beta / N) * Y[2, i - 1] * Y[0, i - 1] - sigma * Y[1, i - 1])
        Y[2, i] = Y[2, i - 1] + dt * (sigma * Y[1, i - 1] - gamma * Y[2, i - 1])
        Y[3, i] = Y[3, i - 1] + dt * (gamma * Y[2, i - 1] - eta * Y[3, i - 1])

    return Y


beta = 0.5
gamma = 0.1
eta = gamma
sigma = gamma
N = 1000
T = 100
dt = 1
x = np.arange(0, T, 1)
I0 = 1


# plot_model(SIR(N, T, dt, gamma, beta, I0), f'Ewolucja SIR dla N={N}, $I_0$={I0}, ' +
#            '$\\beta = ' + f'{beta}$, ' + '$\\gamma = ' + f'{gamma}$, ' +'dt = ' + f'{dt}')
#
# plot_model(SI(N, T, dt, beta, I0), f'Ewolucja SI dla N={N}, $I_0$={I0}, ' +
#            '$\\beta = ' + f'{beta}$, ' + 'dt = ' + f'{dt}')
#
# plot_model(SIS(N, T, dt, gamma, beta, I0), f'Ewolucja SIS dla N={N}, $I_0$={I0}, ' +
#            '$\\beta = ' + f'{beta}$, ' + '$\\gamma = ' + f'{gamma}$, ' + 'dt = ' + f'{dt}')
#
# plot_model(SIRS(N, T, dt, gamma, beta, eta, I0), f'Ewolucja SIRS dla N={N}, $I_0$={I0}, ' +
#            '$\\beta = ' + f'{beta}$, ' + '$\\gamma = ' + f'{gamma}$. ' + '$\\eta = ' + f'{eta}$, ' + 'dt = ' + f'{dt}')

# plot_model(SEIR(N, T, dt, gamma, beta, sigma, I0), f'Ewolucja SEIR dla N={N}, $I_0$={I0}, ' + '$\\beta = ' +
#            f'{beta}$, ' + '$\\gamma = ' + f'{gamma}$, ' + '$\\sigma = ' + f'{sigma}$, ' + 'dt = ' + f'{dt}')

# plot_model(SEIRS(N, T, dt, gamma, beta, sigma, I0), f'Ewolucja SEIRS dla N={N}, $I_0$={I0}, ' +
#            '$\\beta = ' + f'{beta}$ ' + '$\\gamma = ' + f'{gamma}$, ' + '$\\sigma = ' +
#            f'{sigma}$ ' + '$\\eta = ' + f'{eta}$ ' + 'dt=' + f'{dt}')
