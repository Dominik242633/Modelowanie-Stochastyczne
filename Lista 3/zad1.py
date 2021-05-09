import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates

def load_data(file_name):
    x = []

    f = open(file_name, "r")

    for i in f:
        temp = i.split('\t')[:-1]
        x.append(np.array([int(float(temp[0][:5])*1000),   # YYYY
                           int(temp[0][5:7]),            # MM
                           int(temp[0][7:9]),            # DD
                           int(float(temp[1])),     # HH
                           float(temp[2]),           # zonal price, P d,h
                           float(temp[3]),           # system load forecast, Z d,h
                           float(temp[4]),           # zonal load forecast
                           int(float(temp[5]))]))     # day of the week

    f.close()
    x = np.array(x)

    return x


if __name__ == "__main__":
    file_name = "GEFCOM.txt"
    x = load_data(file_name)

    days = [2, 7]
    values = [4, 5]

    for day in days:
        day_x = np.array([i for i in x if i[-1] == day])
        for value in values:
            fig, ax = plt.subplots()

            for rok in sorted(set(day_x[:, 0])):
                x1 = np.array([datetime.datetime(2010, int(i[1]), int(i[2]),
                                                 int(i[3]), 0) for i in day_x if (i[-1] == day and i[0] == rok)])
                y1 = np.array([i[value] for i in day_x if (i[-1] == day and i[0] == rok)])

                ax.plot(x1, y1, '-', label=f'{int(rok)} rok')

            ax.legend(loc='best', frameon=False)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter("%m"))
            plt.title(f"Wykres sezonowy #1 dla dnia {day}")
            plt.xlabel('Miesiąc', fontdict={'size': 16})
            plt.ylabel(f'Wartość {"$P_d$$_,$$_h$" if value==4 else "$Z_d$$_,$$_h$"}', fontdict={'size': 16})
            plt.show()

