import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def init_parameters(I0, N, T):
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


def plot_model(y, title, dt):
    labels = ['$S_t$', '$E_t$', '$I_t$', '$R_t$']

    for i in range(len(y)):
        if np.mean(y[i]) != 0:
            plt.plot(np.arange(0, len(y[i]) * dt, dt), y[i], label=labels[i])
    set_plot_properties(title + '\nSkala liniowa')

    for i in range(len(y)):
        if np.mean(y[i]) != 0:
            plt.plot(np.arange(0, len(y[i]) * dt, dt), y[i], label=labels[i])
    plt.xscale("log")
    plt.yscale("log")
    set_plot_properties(title + '\nSkala logarytmiczna')


def euler_SIR(N, steps, dt, gamma, beta, sigma, eta, I0):
    Y = init_parameters(I0, N, steps)

    for i in range(1, steps):
        Y[0, i] = Y[0, i - 1] + dt * (-(beta / N) * Y[2, i - 1] * Y[0, i - 1])
        Y[2, i] = Y[2, i - 1] + dt * ((beta / N) * Y[2, i - 1] * Y[0, i - 1] - gamma * Y[2, i - 1])
        Y[3, i] = Y[3, i - 1] + dt * (gamma * Y[2, i - 1])

    return Y


def euler_SI(N, steps, dt, gamma, beta, sigma, eta, I0):
    Y = init_parameters(I0, N, steps)

    for i in range(1, steps):
        Y[0, i] = Y[0, i - 1] + dt * (-(beta / N) * Y[2, i - 1] * Y[0, i - 1])
        Y[2, i] = Y[2, i - 1] + dt * ((beta / N) * Y[2, i - 1] * Y[0, i - 1])

    return Y


def euler_SIS(N, steps, dt, gamma, beta, sigma, eta, I0):
    Y = init_parameters(I0, N, steps)

    for i in range(1, steps):
        Y[0, i] = Y[0, i - 1] + dt * (-(beta / N) * Y[2, i - 1] * Y[0, i - 1] + gamma * Y[2, i - 1])
        Y[2, i] = Y[2, i - 1] + dt * ((beta / N) * Y[2, i - 1] * Y[0, i - 1] - gamma * Y[2, i - 1])

    return Y


def euler_SIRS(N, steps, dt, gamma, beta, sigma, eta, I0):
    Y = init_parameters(I0, N, steps)

    for i in range(1, steps):
        Y[0, i] = Y[0, i - 1] + dt * (-(beta / N) * Y[2, i - 1] * Y[0, i - 1] + eta * Y[3, i - 1])
        Y[2, i] = Y[2, i - 1] + dt * ((beta / N) * Y[2, i - 1] * Y[0, i - 1] - gamma * Y[2, i - 1])
        Y[3, i] = Y[3, i - 1] + dt * (gamma * Y[2, i - 1] - eta * Y[3, i - 1])

    return Y


def euler_SEIR(N, steps, dt, gamma, beta, sigma, eta, I0):
    Y = init_parameters(I0, N, steps)

    for i in range(1, steps):
        Y[0, i] = Y[0, i - 1] + dt * (-(beta / N) * Y[2, i - 1] * Y[0, i - 1])
        Y[1, i] = Y[1, i - 1] + dt * ((beta / N) * Y[2, i - 1] * Y[0, i - 1] - sigma * Y[1, i - 1])
        Y[2, i] = Y[2, i - 1] + dt * (sigma * Y[1, i - 1] - gamma * Y[2, i - 1])
        Y[3, i] = Y[3, i - 1] + dt * (gamma * Y[2, i - 1])

    return Y


def euler_SEIRS(N, steps, dt, gamma, beta, sigma, eta, I0):
    Y = init_parameters(I0, N, steps)

    for i in range(1, steps):
        Y[0, i] = Y[0, i - 1] + dt * (-(beta / N) * Y[2, i - 1] * Y[0, i - 1] + eta * Y[3, i - 1])
        Y[1, i] = Y[1, i - 1] + dt * ((beta / N) * Y[2, i - 1] * Y[0, i - 1] - sigma * Y[1, i - 1])
        Y[2, i] = Y[2, i - 1] + dt * (sigma * Y[1, i - 1] - gamma * Y[2, i - 1])
        Y[3, i] = Y[3, i - 1] + dt * (gamma * Y[2, i - 1] - eta * Y[3, i - 1])

    return Y


if __name__ == '__main__':
    beta = 0.5
    gamma = 0.1
    eta = gamma
    sigma = gamma
    N = 1000
    T = 100
    dt = 1
    steps = int(T / dt)
    I0 = 1

    plot_model(euler_SIR(N, steps, dt, gamma, beta, sigma, eta, I0), f'Ewolucja SIR dla N={N}, $I_0$={I0}, ' +
               '$\\beta = ' + f'{beta}$, ' + '$\\gamma = ' + f'{gamma}$, ' +'dt = ' + f'{dt}', dt)

    plot_model(euler_SI(N, steps, dt, gamma, beta, sigma, eta, I0), f'Ewolucja SI dla N={N}, $I_0$={I0}, ' +
               '$\\beta = ' + f'{beta}$, ' + 'dt = ' + f'{dt}', dt)

    plot_model(euler_SIS(N, steps, dt, gamma, beta, sigma, eta, I0), f'Ewolucja SIS dla N={N}, $I_0$={I0}, ' +
               '$\\beta = ' + f'{beta}$, ' + '$\\gamma = ' + f'{gamma}$, ' + 'dt = ' + f'{dt}', dt)

    plot_model(euler_SIRS(N, steps, dt, gamma, beta, sigma, eta, I0), f'Ewolucja SIRS dla N={N}, $I_0$={I0}, ' +
               '$\\beta = ' + f'{beta}$, ' + '$\\gamma = ' + f'{gamma}$. '
               + '$\\eta = ' + f'{eta}$, ' + 'dt = ' + f'{dt}', dt)

    plot_model(euler_SEIR(N, steps, dt, gamma, beta, sigma, eta, I0), f'Ewolucja SEIR dla N={N}, $I_0$={I0}, ' + '$\\beta = ' +
               f'{beta}$, ' + '$\\gamma = ' + f'{gamma}$, ' + '$\\sigma = ' + f'{sigma}$, ' + 'dt = ' + f'{dt}', dt)

    plot_model(euler_SEIRS(N, steps, dt, gamma, beta, sigma, eta, I0), f'Ewolucja SEIRS dla N={N}, $I_0$={I0}, ' +
               '$\\beta = ' + f'{beta}$ ' + '$\\gamma = ' + f'{gamma}$, ' + '$\\sigma = ' +
               f'{sigma}$ ' + '$\\eta = ' + f'{eta}$ ' + 'dt=' + f'{dt}', dt)
