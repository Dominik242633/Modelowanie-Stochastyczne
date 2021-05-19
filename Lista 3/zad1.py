# data[:,2] zonal price, P d,h
# data[:, 3] system load forecast, Z d,h

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


if __name__ == "__main__":
    file_name = "GEFCOM.txt"
    data = np.loadtxt(file_name)

    # Dane zaczynają się od soboty, więc day = 1 będzie niedzielą, day = 3 wtorkiem
    days = [1, 3]
    days_name = ['niedziel', 'wtorków']
    interval = [0.2, 0.8]
    prognosis = ['Cena strefowa', 'Prognoza zapotrzebowania']
    start_day = 2*24

    # Wykres #1 dla sezonowości dobowej
    for index, prognosis_name in enumerate(prognosis):
        for day in days:
            for i in range(1080):
                if (data[i*24][-1] == 6 and day == 1) or (data[i*24][-1] == 1 and day == 3):
                    x = np.arange(0, 24, 1)
                    plt.plot(x, data[i*24 + day*24: i*24 + day*24+24, index + 2])
            plt.title(f'Wykres sezonowy #1 dla {days_name[0] if day == 1 else days_name[1]}')
            plt.xlabel('Godzina')
            plt.ylabel(f'{prognosis_name}')
            plt.show()

    # Wykres #2 dla sezonowości dobowej
    for index, prognosis_name in enumerate(prognosis):
        for day in days:
            for i in range(24):
                x = np.linspace(i + interval[0] - 0.5, i + interval[1] - 0.5, int(len(data[i+day*24::24*7, index + 2])))
                plt.plot(x, data[i+day*24::24*7, index + 2])
                plt.plot(x, np.full((x.shape), np.mean(data[i+day*24::24*7, index + 2])), color='k')
            plt.title(f'Wykres sezonowy #2 dla {days_name[0] if day == 1 else days_name[1]}')
            plt.xlabel('Godzina')
            plt.ylabel(f'{prognosis_name}')
            plt.show()

    # Wykres #1 dla sezonowości tygodniowej
    for index, prognosis_name in enumerate(prognosis):
        for week in range(int((data.shape[0]/24)/7)):
            y = []
            for day in range(7):
                y.append(np.mean(data[week*7*24 + day*24 + start_day: week*7*24 + day*24+24 + start_day, index + 2]))
            x = np.arange(0, 7, 1)
            plt.plot(x, y)
        plt.title(f'Wykres sezonowy #1 dla tygodni')
        plt.xlabel('Dzień tygodnia')
        plt.ylabel(f'{prognosis_name}')
        plt.show()

    # Wykres #2 dla sezonowości tygodniowej
    for index, prognosis_name in enumerate(prognosis):
        for i in range(7):
            y = []
            for week in range(int((data.shape[0]/24)/7)):
                y.append(np.mean(data[week*7*24 + i*24 + start_day: week*7*24 + i*24+24 + start_day, index + 2]))
            x = np.linspace(i + interval[0] - 0.5, i + interval[1] - 0.5, int((data.shape[0]/24)/7))
            plt.plot(x, y)
            plt.plot(x, np.full((len(y)), np.mean(np.array(y))), color='k')
        plt.title(f'Wykres sezonowy #2 dla tygodni')
        plt.xlabel('Dzień tygodnia')
        plt.ylabel(f'{prognosis_name}')
        plt.show()
