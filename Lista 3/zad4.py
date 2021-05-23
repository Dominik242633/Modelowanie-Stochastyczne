import numpy as np
import matplotlib.pyplot as plt
from holtwinters import holtwinters
from scipy.optimize import minimize
from zad5 import print_rate, plot_prognosis, plot_score
plt.style.use('ggplot')


def naive_prediction(actual):
    prediction = []
    for i in range(360*24, actual.shape[0]):
        if actual[i, -1] in [1, 6, 7]:
            prediction.append(actual[i-7*24, 2])
        else:
            prediction.append(actual[i-24, 2])

    return prediction


def hw_method(p, s=7, T=360, initial_param=np.array([.5, .5, .5])):
    """p - all data from one hour
        s - period(weekly, i.e., 7 day, for daily data)
        T - Last day of the calibration period
        initial_param - [alfa, beta, gamma] - Holt-Winters method parameters"""
    param = minimize(holtwinters, initial_param, args=(s, p[:T])).x
    pf = holtwinters(param, s, p, return_fx=True)
    return pf[T:]


if __name__ == "__main__":
    file_name = "GEFCOM.txt"
    data = np.loadtxt(file_name)

    actual = data[:, 2]
    y_naive_prediction = np.array(naive_prediction(data))

    print_rate(y_naive_prediction, actual[360 * 24:], "Naive method")
    plot_prognosis(y_naive_prediction, actual, "Naive method")
    plot_score(y_naive_prediction, actual[360 * 24:], 'metody naiwnej')

    hw_prediction = np.zeros(data.shape)

    for hour in range(24):
        hw_prediction[360*24+hour::24, 2] = hw_method(data[hour::24, 2])

    print_rate(hw_prediction[360 * 24:, 2], data[360 * 24:, 2], "Holt-Winters method")
    plot_prognosis(hw_prediction[360*24:, 2], actual, "Holt-Winters method")
    plot_score(hw_prediction[360 * 24:, 2], actual[360 * 24:], 'metody Holta-Wintersa')
