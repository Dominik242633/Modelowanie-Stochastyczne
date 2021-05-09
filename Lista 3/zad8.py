import numpy as np
import matplotlib.pyplot as plt
from epf_arx import epf_arx
from zad5 import print_rate, plot_prognosis, predict
import zad6
import zad7
plt.style.use('ggplot')


def arx_const_window(actual_data):
    startd = 0
    endd = 360
    Ndays = 776
    res = epf_arx(actual_data[:, :4], Ndays, startd, endd, 'arx')

    return res[:, 3]


def arx_expandable_window(actual_data):
    res = []
    startd = 0
    endd = 360
    Ndays = 776
    for i in range(Ndays):
        res = res + list(epf_arx(actual_data[:, :4], 1, startd, endd, 'arx')[:, 3])
        endd += 1

    return res


def arx_rolled_window(actual_data):
    res = []
    startd = 0
    endd = 360
    Ndays = 776
    for i in range(Ndays):
        res = res + list(epf_arx(actual_data[:, :4], 1, startd, endd, 'arx')[:, 3])
        startd += 1
        endd += 1

    return res


if __name__ == "__main__":
    data = np.loadtxt('NPdata_2013-2016.txt')
    actual = data[:, 2]

    prediction = predict(data, Ndays=776)
    plot_prognosis(prediction, actual, 'stałego, 360-dniowego okna kalibracji')
    print_rate(prediction, actual[360*24:], 'ARX - 360-dniowe okno kalibracji')

    prediction = zad6.predict(data, Ndays=776)
    plot_prognosis(prediction, actual, 'rozszerzanego okna kalibracji')
    print_rate(prediction, actual[360*24:], 'ARX - rozszerzane okno kalibracji')

    prediction = zad7.predict(data, Ndays=776)
    plot_prognosis(prediction, actual, 'rolowanego okna kalibracji')
    print_rate(prediction, actual[360*24:], 'ARX - rolowane okno kalibracji')
