# float(temp[2]),  # zonal price, P d,h
# float(temp[3]),  # system load forecast, Z d,h

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


if __name__ == "__main__":
    file_name = "GEFCOM.txt"
    data = np.loadtxt(file_name)

    days = [1, 6]
    days_name = ['wtorków', 'niedziel']
    interval = [0.2, 0.8]
    prognosis = ['Cena strefowa', 'Prognoza zapotrzebowania']

    for index, prognosis_name in enumerate(prognosis):
        for day in days:
            for i in range(24):
                x = np.linspace(i + interval[0], i + interval[1], int(len(data[i+day*24::24*7, index + 2])))
                plt.plot(x, data[i+day*24::24*7, index + 2])
                plt.plot(x, np.full((x.shape), np.mean(data[i+day*24::24*7, index + 2])), color='k')
            plt.title(f'Wykres sezonowy #1 dla {days_name[0] if day == 1 else days_name[1]}')
            plt.xlabel('Godzina')
            plt.ylabel(f'{prognosis_name}')
            plt.show()
