import numpy as np
from scipy.integrate import RK45
import matplotlib.pyplot as plt


def set_plot_properties(title):
    plt.title(title)
    plt.legend(loc='best', frameon=False)
    plt.xlabel('Czas', fontdict={'size': 16})
    plt.ylabel('Liczba przypadków', fontdict={'size': 16})
    plt.show()


def plot_model(y, title):
    labels = ['$S_t^{DOPRI}$', '$E_t^{DOPRI}$', '$I_t^{DOPRI}$', '$R_t^{DOPRI}$']

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


def SIR_ode(t, y, params):
    beta, gamma, eta, sigma, N = params
    S0, E0, I0, R0 = y
    S1 = -(beta / N) * I0 * S0
    I1 = (beta / N) * I0 * S0 - gamma * I0
    R1 = gamma * I0
    return [S1, E0, I1, R1]


def SI_ode(t, y, params):
    beta, gamma, eta, sigma, N = params
    S0, E0, I0, R0 = y
    S1 = -(beta / N) * I0 * S0
    I1 = (beta / N) * I0 * S0
    return [S1, E0, I1, R0]


def SIS_ode(t, y, params):
    beta, gamma, eta, sigma, N = params
    S0, E0, I0, R0 = y
    S1 = -(beta / N) * I0 * S0 + gamma * I0
    I1 = (beta / N) * I0 * S0 - gamma * I0
    return [S1, E0, I1, R0]


def SIRS_ode(t, y, params):
    beta, gamma, eta, sigma, N = params
    S0, E0, I0, R0 = y
    S1 = -(beta / N) * I0 * S0 + eta * R0
    I1 = (beta / N) * I0 * S0 - gamma * I0
    R1 = gamma * I0 - eta * R0
    return [S1, E0, I1, R1]


def SEIR_ode(t, y, params):
    beta, gamma, eta, sigma, N = params
    S0, E0, I0, R0 = y
    S1 = -(beta / N) * I0 * S0
    E1 = (beta / N) * I0 * S0 - sigma * E0
    I1 = sigma * E0 - gamma * I0
    R1 = gamma * I0
    return [S1, E1, I1, R1]


def SEIRS_ode(t, y, params):
    beta, gamma, eta, sigma, N = params
    S0, E0, I0, R0 = y
    S1 = -(beta / N) * I0 * S0 + eta * R0
    E1 = (beta / N) * I0 * S0 - sigma * E0
    I1 = sigma * E0 - gamma * I0
    R1 = gamma * I0 - eta * R0
    return [S1, E1, I1, R1]


def solve(func, I_0, N, dt, steps, beta, gamma, eta, sigma):
    T_ode = np.zeros(steps)
    Y = np.zeros((4, steps))
    Y[:, 0] = [N - I_0, 0, I_0, 0]

    for T in range(steps - 1):
        ode_system = RK45(lambda t, y: func(t, y, [beta, gamma, eta, sigma, N]), T_ode[T], Y[:, T], T_ode[T] + dt)
        while ode_system.status == 'running':
            ode_system.step()
        Y[:, T + 1] = ode_system.y
        T_ode[T + 1] = ode_system.t

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
    I_0 = 1

    # plot_model(solve(SIR_ode, I_0, N, dt, steps, beta, gamma, eta, sigma), f'Ewolucja SIR dla N={N}, $I_0$={I_0}, ' +
    #            '$\\beta = ' + f'{beta}$, ' + '$\\gamma = ' + f'{gamma}$, ' + 'dt = ' + f'{dt}')

    plot_model(solve(SI_ode, I_0, N, dt, steps, beta, gamma, eta, sigma), f'Ewolucja SI dla N={N}, $I_0$={I_0}, ' +
               '$\\beta = ' + f'{beta}$, ' + 'dt = ' + f'{dt}')

    # plot_model(solve(SIS_ode, I_0, N, dt, steps, beta, gamma, eta, sigma), f'Ewolucja SIS dla N={N}, $I_0$={I_0}, ' +
    #            '$\\beta = ' + f'{beta}$, ' + '$\\gamma = ' + f'{gamma}$, ' + 'dt = ' + f'{dt}')
    #
    # plot_model(solve(SIRS_ode, I_0, N, dt, steps, beta, gamma, eta, sigma), f'Ewolucja SIRS dla N={N}, $I_0$={I_0}, ' +
    #            '$\\beta = ' + f'{beta}$, ' + '$\\gamma = ' + f'{gamma}$, ' + '$\\eta = ' + f'{eta}$, ' + 'dt = ' + f'{dt}')
    #
    # plot_model(solve(SEIR_ode, I_0, N, dt, steps, beta, gamma, eta, sigma), f'Ewolucja SEIR dla N={N}, $I_0$={I_0}, ' +
    #            '$\\beta = ' + f'{beta}$, ' + '$\\gamma = ' + f'{gamma}$, ' + '$\\sigma = ' + f'{sigma}$, ' + 'dt = ' + f'{dt}')
    #
    # plot_model(solve(SEIRS_ode, I_0, N, dt, steps, beta, gamma, eta, sigma), f'Ewolucja SEIRS dla N={N}, $I_0$={I_0}, ' +
    #            '$\\beta = ' + f'{beta}$ ' + '$\\gamma = ' + f'{gamma}$, ' + '$\\sigma = ' +
    #            f'{sigma}$ ' + '$\\eta = ' + f'{eta}$ ' + 'dt=' + f'{dt}')

