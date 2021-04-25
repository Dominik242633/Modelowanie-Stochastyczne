import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from scipy import stats
plt.style.use('ggplot')


gamma = 0.25
country = 'Poland'

data_confirmed = pd.read_csv("confirmed.csv")
data_deaths = pd.read_csv("deaths.csv")
data_recovered = pd.read_csv("recovered.csv")

country_data_confirmed = np.array(data_confirmed.loc[data_confirmed['Country/Region'] == country].values.tolist()[0][4:])
country_data_deaths = np.array(data_deaths.loc[data_deaths['Country/Region'] == country].values.tolist()[0][4:])
country_data_recovered = np.array(data_recovered.loc[data_recovered['Country/Region'] == country].values.tolist()[0][4:])

x = list(data_confirmed)[4:]
y = country_data_confirmed - country_data_deaths - country_data_recovered

# x[42] - 4 marca 2020
# x[425] - 22 marca 2021
start_number = 42
R0 = np.zeros(len(x) - start_number)

for i in range(start_number, len(x)):
    alfa, intercept, r, p, std_err = stats.linregress(np.arange(0, len(x[i - 7:i])), np.log(y[i - 7:i]))
    R0[i - start_number] = (alfa / gamma) + 1

plt.plot_date(dates.datestr2num(x[start_number:len(x)]), R0, '-', label='$R_0$')
plt.title('Estymacja $R_0$')
plt.legend(loc='best', frameon=False)
plt.xlabel('Czas', fontdict={'size': 16})
plt.ylabel('Wartość', fontdict={'size': 16})
plt.xticks(rotation='vertical')
plt.show()
