import numpy as np
import matplotlib.pyplot as plt
from holtwinters import holtwinters
from scipy.optimize import minimize
from zad5 import print_rate, plot_prognosis
plt.style.use('ggplot')


def naive_prediction(actual):
    prediction = []
    for i in range(360*24, len(actual)):
        if actual[-1] in [1, 6, 7]:
            prediction.append(actual[i-7*24])
        else:
            prediction.append(actual[i-24])

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

    actual = np.array([i[2] for i in data[:]])
    y_naive_prediction = np.array(naive_prediction(actual))

    print_rate(y_naive_prediction, actual[360 * 24:], "Naive method")
    plot_prognosis(y_naive_prediction, actual, "Naive method")

    hour = 8
    hw_prediction = hw_method(data[hour::24, 2])

    plt.plot(np.arange(0, len(actual), 1), actual, label='actual')
    plt.plot(np.arange(361*24, len(hw_prediction)*24 + 361*24, 24), hw_prediction, label='prediction')
    plt.title('Predykcja metodą Holta-Wintersa')
    plt.xlabel('Czas')
    plt.ylabel('Wartość')
    plt.legend(loc='best', frameon=False)
    plt.show()

    print_rate(hw_prediction, data[hour::24, 2][360:], "Holt-Winters method")
