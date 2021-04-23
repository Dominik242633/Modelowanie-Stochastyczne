import numpy as np
import matplotlib.pyplot as plt
from zad1 import euler_SI, euler_SIS, euler_SIRS
from zad2 import SI_ode, SIS_ode, SIRS_ode, solve
plt.style.use('ggplot')


def plot_different_dt(euler_func, ode_func, N, T, dt, gamma, beta, eta, I0, title):
    euler_labels = ['$S_t$', '$E_t$', '$I_t$', '$R_t$']
    ode_labels = ['$S_t^{DOPRI}$', '$E_t^{DOPRI}$', '$I_t^{DOPRI}$', '$R_t^{DOPRI}$']

    ode_y = solve(ode_func, I0, N, 1, 100, beta, gamma, eta, sigma)

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    fig.suptitle(title)

    for j, ax in enumerate(axs.flat):
        euler_y = euler_func(N, int(T / dt[j]), dt[j], gamma, beta, sigma, eta, I0)
        for i in range(len(euler_y)):
            if np.mean(euler_y[i]) != 0:
                ax.plot(np.arange(0, len(euler_y[i]) * dt[j], dt[j]), euler_y[i], label=euler_labels[i])
                ax.plot(np.arange(len(ode_y[i])), ode_y[i], label=ode_labels[i])
                ax.set_title(f'dt = {dt[j]}')

    plt.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.4, 1.4))
    fig.tight_layout()
    plt.show()


beta = 0.5
gamma = 0.1
eta = gamma
sigma = gamma
N = 1000
T = 100
dt = [1, 0.1, 0.01, 0.001]
# steps = int(T / dt)
I0 = 1

plot_different_dt(euler_SI, SI_ode, N, T, dt, gamma, beta, eta, I0, 'Porównanie modelu SI dla różnych dt')
plot_different_dt(euler_SIS, SIS_ode, N, T, dt, gamma, beta, eta, I0, 'Porównanie modelu SIS dla różnych dt')
plot_different_dt(euler_SIRS, SIRS_ode, N, T, dt, gamma, beta, eta, I0, 'Porównanie modelu SIRS dla różnych dt')
