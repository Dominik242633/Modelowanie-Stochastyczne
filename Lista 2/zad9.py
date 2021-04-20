import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from scipy import stats
from scipy.integrate import RK45


def set_plot_properties(title, ylabel):
    plt.title(title)
    plt.legend(loc='best', frameon=False)
    plt.xlabel('Czas', fontdict={'size': 16})
    plt.ylabel(ylabel, fontdict={'size': 16})
    plt.xticks(rotation='vertical')



def plot_model(x, y, beta, title):

    for i in range(len(y)):
        plt.plot_date(x, y[i], '-', label='$I_t^{DOPRI}$,' + f' $\\beta_{i + 1} = {round(beta[i], 2)}$')
        set_plot_properties(title + '\nSkala liniowa', ylabel='Liczba przypadków')
    plt.show()

    for i in range(len(y)):
        plt.plot_date(x, np.log(y[i]), '-', label='$I_t^{DOPRI}$,' + f' $\\beta_{i + 1} = {round(beta[i], 2)}$')
        set_plot_properties(title + '\nSkala logarytmiczna', ylabel='Log(Liczba przypadków)')
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


def plot_return_linregress(x, y):
    slope = []

    for i in range(1, len(x)):
        alfa, intercept, r, p, std_err = stats.linregress(np.arange(0, len(x[i])), np.log(y[i]))
        plt.plot_date(dates.datestr2num(x[i]), intercept + alfa * np.arange(0, len(x[i])), '-', linewidth=4, label=f'$\\beta_{i}$')
        slope.append(alfa)
    set_plot_properties('Estymacja $\\beta$ dla ' + f'{country}', ylabel='Log(Liczba przypadków)')
    plt.plot_date(dates.datestr2num(x[0]), np.log(y[0]), 'ko', markersize=4)
    plt.show()

    return slope


N = 38000000
start = 46
T = 340 - start
dt = 1
steps = int(T / dt)
I_0 = 1
gamma = 0.25
country = 'Poland'

data_confirmed = pd.read_csv("confirmed.csv")
data_deaths = pd.read_csv("deaths.csv")
data_recovered = pd.read_csv("recovered.csv")

country_data_confirmed = data_confirmed.loc[data_confirmed['Country/Region'] == country].to_dict('list')
country_data_deaths = data_deaths.loc[data_deaths['Country/Region'] == country].to_dict('list')
country_data_recovered = data_recovered.loc[data_recovered['Country/Region'] == country].to_dict('list')

x = []
y1 = []
x2 = []
y2 = []
x3 = []
y3 = []

for key, value in country_data_confirmed.items():
    x.append(key)
    y1.append(value[0])

for value in country_data_deaths.values():
    y2.append(value[0])

for value in country_data_recovered.values():
    y3.append(value[0])

x = x[4:344]
y1 = np.array(y1[4:])
y2 = np.array(y2[4:])
y3 = np.array(y3[4:])
y = y1 - y2 - y3

# Estymacja parametrów na podstawie danych z 8 marca - 9 kwietnia 2020
x0 = x[43:79]
y0 = y[43:79]
x1 = x[46:54]
y1 = y[46:54]
x2 = x[57:66]
y2 = y[57:66]
x3 = x[71:79]
y3 = y[71:79]

# Estymacja parametrów na podstawie danych z 10 - 30 kwietnia 2020
# x0 = x[76:100]
# y0 = y[76:100]
# x1 = x[79:86]
# y1 = y[79:86]
# x2 = x[86:93]
# y2 = y[86:93]
# x3 = x[93:100]
# y3 = y[93:100]

alfa1, alfa2, alfa3 = plot_return_linregress([x0, x1, x2, x3], [y0, y1, y2, y3])
beta = np.array([alfa1, alfa2, alfa3]) + gamma

plot_model(dates.datestr2num(x[start:]),
           [solve(SIR_ode, I_0, N, dt, steps, beta[0], gamma)[2],
               solve(SIR_ode, I_0, N, dt, steps, beta[1], gamma)[2],
               solve(SIR_ode, I_0, N, dt, steps, beta[2], gamma)[2]],
           beta, "Prognoza zarażonych w " + f'{country}')
