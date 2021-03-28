import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from scipy import stats
from scipy.integrate import RK45


def set_plot_properties(title):
    plt.title(title)
    plt.legend(loc='best', frameon=False)
    plt.xlabel('Czas', fontdict={'size': 16})
    plt.ylabel('Liczba przypadków', fontdict={'size': 16})


def plot_model(x, y, beta, title):

    for i in range(len(y)):
        plt.plot_date(x, y[i], '-', label='$I_t^{DOPRI}$,' + ' $\\gamma = $' + f'{round(beta[i], 2)}')
        set_plot_properties(title + '\nSkala liniowa')
    plt.show()

    for i in range(len(y)):
        plt.plot_date(x, y[i], '-', label='$I_t^{DOPRI}$,' + ' $\\gamma = $' + f'{round(beta[i], 2)}')
        plt.yscale("log")
        set_plot_properties(title + '\nSkala logarytmiczna')
    plt.show()


def SIR_ode(t, y, params):
    beta, gamma, N = params
    S0, E0, I0, R0 = y
    S1 = -(beta / N) * I0 * S0
    I1 = (beta / N) * I0 * S0 - gamma * I0
    R1 = gamma * I0
    return [S1, E0, I1, R1]


def solve(func, I_0, N, dt, steps, beta, gamma):
    T_ode = np.zeros(steps)
    Y = np.zeros((4, steps))
    Y[:, 0] = [N - I_0, 0, I_0, 0]

    for T in range(steps - 1):
        ode_system = RK45(lambda t, y: func(t, y, [beta, gamma, N]), T_ode[T], Y[:, T], T_ode[T] + dt)
        while ode_system.status == 'running':
            ode_system.step()
        Y[:, T + 1] = ode_system.y
        T_ode[T + 1] = ode_system.t

    return Y


data = pd.read_csv("time_series_covid19_confirmed_global.csv")

z = data.loc[data['Country/Region'] == 'Poland'].to_dict('list')

x = []
y = []
gamma = 0.25

for key, value in z.items():
    x.append(key)
    y.append(value[0])

x = x[4:344]
y = np.array(y[4:], dtype='int')

x0 = x[43:79]
y0 = y[43:79]
x1 = x[46:54]
y1 = y[46:54]
x2 = x[57:66]
y2 = y[57:66]
x3 = x[71:79]
y3 = y[71:79]

alfa1, intercept1, r1, p1, std_err1 = stats.linregress(np.arange(0, len(x1)), np.log(y1))
alfa2, intercept2, r2, p2, std_err2 = stats.linregress(np.arange(0, len(x2)), np.log(y2))
alfa3, intercept3, r3, p3, std_err3 = stats.linregress(np.arange(0, len(x3)), np.log(y3))

x = dates.datestr2num(x)
x0 = dates.datestr2num(x0)
x1 = dates.datestr2num(x1)
x2 = dates.datestr2num(x2)
x3 = dates.datestr2num(x3)


plt.plot_date(x0, np.log(y0))
plt.plot_date(x1, intercept1 + alfa1 * np.arange(0, len(x1)), 'b-')
plt.plot_date(x2, intercept2 + alfa2 * np.arange(0, len(x2)), 'r-')
plt.plot_date(x3, intercept3 + alfa3 * np.arange(0, len(x3)), 'y-')
plt.show()

beta1 = alfa1 + gamma
beta2 = alfa2 + gamma
beta3 = alfa3 + gamma

N = 38000000
T = 340
dt = 1
steps = int(T / dt)
I_0 = 1

y1 = solve(SIR_ode, I_0, N, dt, steps, beta1, gamma)[2]
y2 = solve(SIR_ode, I_0, N, dt, steps, beta2, gamma)[2]
y3 = solve(SIR_ode, I_0, N, dt, steps, beta3, gamma)[2]

plot_model(x, [y1, y2, y3], [beta1, beta2, beta3], "Prognoza zarażonych w Polsce")
