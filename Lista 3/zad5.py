import numpy as np
import matplotlib.pyplot as plt
from epf_arx import epf_arx
plt.style.use('ggplot')


def print_rate(prediction, actual_data, method_name):
    print(method_name)
    print('MAE = ' + str(np.mean(np.abs(np.array(prediction) - np.array(actual_data)))))
    print('RMSE = ' + str(np.sqrt(np.mean((np.array(prediction) - np.array(actual_data))**2))))


def plot_prognosis(prediction, actual_data, title):
    plt.plot(np.arange(0, len(actual_data), 1), actual_data, label='actual')
    plt.plot(np.arange(360 * 24, len(prediction) + 360 * 24, 1), prediction, label='predicted')
    plt.title(f'Predykcja metodą ARX dla {title}')
    plt.xlabel('Czas')
    plt.ylabel('Wartość')
    plt.legend(loc='best', frameon=False)
    plt.show()


def predict(actual_data, startd=0, endd=360, Ndays=722):
    result = epf_arx(actual_data[:, :4], Ndays, startd, endd, 'arx', 'constant')[:, 3]
    return result


if __name__ == "__main__":
    data = np.loadtxt('GEFCOM.txt')

    actual = np.array([i[2] for i in data[:]])
    predicted = np.array(predict(data))

    print_rate(predicted, actual[360*24:], "ARX - 360-dniowe okno kalibracji")

    plot_prognosis(predicted, actual, 'dla stałego, 360-dniowego okna')
