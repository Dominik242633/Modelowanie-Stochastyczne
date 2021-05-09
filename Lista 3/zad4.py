from zad1 import load_data
import numpy as np
import matplotlib.pyplot as plt
from holtwinters import holtwinters
from scipy.optimize import minimize
plt.style.use('ggplot')


def mae_function(prediction: list, actual: list) -> float:
    result = np.mean(np.abs(np.array(prediction) - np.array(actual)))
    return result


def rmse_function(prediction: list, actual: list) -> float:
    result = np.sqrt(np.mean((np.array(prediction) - np.array(actual))**2))
    return result


def naive_prediction(data: list) -> list:
    """data - list of lists
        return list of predicted values"""
    prediction = [i[4] for i in data[:361*24]]
    for i in range(361*24, len(data)):
        if data[i][-1] in [1, 6, 7]:
            prediction.append(data[i-7*24][4])
        else:
            prediction.append(data[i-24][4])

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
    x = load_data(file_name)
    y_naive_prediction = naive_prediction(x)

    plt.plot(np.arange(0, len(x[:]), 1), [i[4] for i in x[:]], label='actual')
    plt.plot(np.arange(361*24, len(y_naive_prediction[361*24:]) + 361*24, 1),
             y_naive_prediction[361*24:], label='prediction')
    plt.title('Predykcja metodą naiwną')
    plt.xlabel('Czas')
    plt.ylabel('Wartość')
    plt.legend(loc='best', frameon=False)
    plt.show()

    d = np.loadtxt('GEFCOM.txt')
    hour = 8
    hw_prediction = hw_method(d[hour::24, 2])

    plt.plot(np.arange(0, len(x[:]), 1), [i[4] for i in x[:]], label='actual')
    plt.plot(np.arange(361*24, len(hw_prediction)*24 + 361*24, 24), hw_prediction, label='prediction')
    plt.title('Predykcja metodą Holta-Wintersa')
    plt.xlabel('Czas')
    plt.ylabel('Wartość')
    plt.legend(loc='best', frameon=False)
    plt.show()

    print("Naive method MAE = " + str(mae_function(y_naive_prediction[361 * 24:], [i[4] for i in x[361 * 24:]])))
    print("Naive method RMSE = " + str(rmse_function(y_naive_prediction[361 * 24:], [i[4] for i in x[361 * 24:]])))

    print("Holt-Winters method MAE = " + str(mae_function(d[hour::24, 2][360:], hw_prediction)))
    print("Holt-Winters method RMSE = " + str(rmse_function(d[hour::24, 2][360:], hw_prediction)))
