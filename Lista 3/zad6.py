import numpy as np
import matplotlib.pyplot as plt
from epf_arx import epf_arx
from zad5 import print_rate, plot_prognosis
plt.style.use('ggplot')


def predict(actual_data, startd=0, endd=360, Ndays=722):
    res = []
    for i in range(Ndays):
        res = res + list(epf_arx(actual_data[:, :4], 1, startd, endd, 'arx')[:, 3])
        endd += 1
    return res


if __name__ == "__main__":
    data = np.loadtxt('GEFCOM.txt')

    actual = np.array([i[2] for i in data[:]])
    predicted = np.array(predict(data))

    print_rate(predicted, actual[360*24:], "ARX - rozszerzane okno kalibracji")

    plot_prognosis(predicted, actual, 'dla rozszerzanego okna kalibracji')
