import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from scipy import stats
plt.style.use('ggplot')


N = 38000000
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
y2 = []
y3 = []

for key, value in country_data_confirmed.items():
    x.append(key)
    y1.append(value[0])

for value in country_data_deaths.values():
    y2.append(value[0])

for value in country_data_recovered.values():
    y3.append(value[0])

x = x[4:]
y1 = np.array(y1[4:])
y2 = np.array(y2[4:])
y3 = np.array(y3[4:])
y = y1 - y2 - y3

# x[42] - 4 marca 2020
# x[425] - 22 marca 2021
start_number = 42
R0 = np.zeros(len(x) - 1 - start_number)

for i in range(start_number, len(x) - 1):
    alfa, intercept, r, p, std_err = stats.linregress(np.arange(0, len(x[i - 7:i])), np.log(y[i - 7:i]))
    R0[i - start_number] = (alfa / gamma) + 1

plt.plot_date(dates.datestr2num(x[start_number:len(x) - 1]), R0, '-', label='$R_0$')
plt.title('Estymacja $R_0$')
plt.legend(loc='best', frameon=False)
plt.xlabel('Czas', fontdict={'size': 16})
plt.ylabel('Wartość', fontdict={'size': 16})
plt.xticks(rotation='vertical')
plt.show()
