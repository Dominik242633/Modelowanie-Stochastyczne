import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from zad4 import naive_prediction, hw_method
from zad5 import print_rate, plot_prognosis, predict, plot_score
import zad6
import zad7
plt.style.use('ggplot')


if __name__ == "__main__":
    data = np.loadtxt('NPdata_2013-2016.txt')
    actual = data[:, 2]

    weekday = np.array([datetime.strptime(str(int(data[i, 0])), "%Y%m%d").weekday() + 1 for i in range(data.shape[0])])
    weekday = np.reshape(weekday, (weekday.shape[0], 1))
    actual_naive = np.hstack((data, weekday))
    y_naive_prediction = np.array(naive_prediction(actual_naive))

    print_rate(y_naive_prediction, actual[360 * 24:], "Naive method")
    plot_prognosis(y_naive_prediction, actual, "Naive method")
    plot_score(y_naive_prediction, actual[360*24:], 'metody naiwnej')

    hw_prediction = np.zeros(data.shape)

    for hour in range(24):
        hw_prediction[360*24+hour::24, 2] = hw_method(data[hour::24, 2])

    print_rate(hw_prediction[360 * 24:, 2], data[360 * 24:, 2], "Holt-Winters method")
    plot_prognosis(hw_prediction[360*24:, 2], actual, "Holt-Winters method")
    plot_score(hw_prediction[360 * 24:, 2], actual[360*24:], 'metody Holta-Wintersa')

    prediction = predict(data, Ndays=776)
    plot_prognosis(prediction, actual, 'stałego, 360-dniowego okna kalibracji')
    print_rate(prediction, actual[360*24:], 'ARX - 360-dniowe okno kalibracji')
    plot_score(prediction, actual[360*24:], 'ARX z stałym 360-dniowym oknem')

    prediction = zad6.predict(data, Ndays=776)
    plot_prognosis(prediction, actual, 'rozszerzanego okna kalibracji')
    print_rate(prediction, actual[360*24:], 'ARX - rozszerzane okno kalibracji')
    plot_score(prediction, actual[360*24:], 'ARX z rozszerzanym oknem')

    prediction = zad7.predict(data, Ndays=776)
    plot_prognosis(prediction, actual, 'rolowanego okna kalibracji')
    print_rate(prediction, actual[360*24:], 'ARX - rolowane okno kalibracji')
    plot_score(prediction, actual[360*24:], 'ARX z rolowanym oknem')
