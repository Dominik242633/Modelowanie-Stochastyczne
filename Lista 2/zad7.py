import numpy as np
import matplotlib.pyplot as plt
from zad5 import stochastic_SIR, stochastic_SI, stochastic_SIS, stochastic_SIRS, \
    stochastic_SEIR, stochastic_SEIRS, init_parameters
plt.style.use('ggplot')


beta = 0.5
gamma = 0.1
eta = gamma
sigma = gamma
N = 1000
T = 100 * N
dt = 1/N
I_0 = 1
quantille = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89]
colors = [(1, 0.56, 0, 1), (1, 0.44, 0, 1), (1, 0.33, 0, 1), (1, 0.22, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1),
          (1, 0.22, 0, 1), (1, 0.33, 0, 1), (1, 0.44, 0, 1), (1, 0.56, 0, 1)]
trajectories_number = 100


def get_values(stochastic_model):
    S_values = np.zeros((trajectories_number, T))
    E_values = np.zeros((trajectories_number, T))
    I_values = np.zeros((trajectories_number, T))
    R_values = np.zeros((trajectories_number, T))

    for i in range(trajectories_number):
        print(f'Trajectory number: {i}')
        a = stochastic_model(init_parameters(N, T, I_0), beta, gamma, eta, sigma)
        for j in range(len(a)):
            a[j] = np.array(a[j])
            a[j] = a[j].flatten()

        S_values[i], E_values[i], I_values[i], R_values[i] = a

    values = [np.sort(S_values, axis=0),
                np.sort(E_values, axis=0),
                np.sort(I_values, axis=0),
                np.sort(R_values, axis=0)]

    return values


def fan_chart(values, title):
    labels = ['$S_t$', '$E_t$', '$I_t$', '$R_t$']

    for i in range(len(values)):
        if np.mean(values[i][-1]) != 0:
            for index, element in enumerate(quantille):
                plt.fill_between(np.arange(0, len(values[i][element]) * dt, dt),
                                 values[i][element], values[i][element+10], color=colors[index])
            plt.title(title + labels[i])
            plt.xlabel('Czas', fontdict={'size': 16})
            plt.ylabel('Liczba przypadków', fontdict={'size': 16})
            plt.show()


fan_chart(get_values(stochastic_SIR), 'Model SIR, dla ')

fan_chart(get_values(stochastic_SI), 'Model SI, dla ')

fan_chart(get_values(stochastic_SIS), 'Model SIS, dla ')

fan_chart(get_values(stochastic_SIRS), 'Model SIRS, dla ')

fan_chart(get_values(stochastic_SEIR), 'Model SEIR, dla ')

fan_chart(get_values(stochastic_SEIRS), 'Model SEIRS, dla ')
