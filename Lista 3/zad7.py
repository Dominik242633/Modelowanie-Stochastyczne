import numpy as np
import matplotlib.pyplot as plt
from epf_arx import epf_arx
from zad5 import print_rate, plot_prognosis, plot_score
plt.style.use('ggplot')


def predict(actual_data, startd=0, endd=360, Ndays=722):
    result = epf_arx(actual_data[:, :4], Ndays, startd, endd, 'arx', 'rolled')[:, 3]
    return result


if __name__ == "__main__":
    data = np.loadtxt('GEFCOM.txt')

    actual = np.array([i[2] for i in data[:]])
    predicted = np.array(predict(data))

    print_rate(predicted, actual[360*24:], "ARX - rolowane okno kalibracji")

    plot_prognosis(predicted, actual, 'dla rolowanego okna kalibracji')

    plot_score(predicted, actual[360*24:], 'ARX z rolowanym oknem')
