import numpy as np
import matplotlib.pyplot as plt
from epf_arx import epf_arx
plt.style.use('ggplot')


def print_rate(prediction, actual_data, method_name):
    print(method_name)
    print('MAE = ' + str(np.mean(np.abs(np.array(prediction) - np.array(actual_data)))))
    print('RMSE = ' + str(np.sqrt(np.mean((np.array(prediction) - np.array(actual_data))**2))))


def set_plot_properties(title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best', frameon=False)
    plt.show()


def plot_prognosis(prediction, actual_data, title):
    plt.plot(np.arange(0, len(actual_data), 1), actual_data, label='actual')
    plt.plot(np.arange(360 * 24, len(prediction) + 360 * 24, 1), prediction, label='predicted')
    set_plot_properties(f'Predykcja metodą ARX dla {title}', 'Czas', 'Wartość')


def plot_score(prediction, actual, prognosis_name):
    x = np.arange(0, 24, 1)
    mae = [np.mean(np.abs(np.array(prediction[i::24]) - np.array(actual[i::24]))) for i in range(24)]
    rmse = [np.sqrt(np.mean((np.array(prediction[::24]) - np.array(actual[i::24]))**2)) for i in range(24)]
    plt.plot(x, mae, label='MAE')
    plt.plot(x, rmse, label='RMSE')
    set_plot_properties(f'MAE i RMSE dla {prognosis_name}', 'Godzina', 'Wartość')


def predict(actual_data, startd=0, endd=360, Ndays=722):
    result = epf_arx(actual_data[:, :4], Ndays, startd, endd, 'arx', 'constant')[:, 3]
    return result


if __name__ == "__main__":
    data = np.loadtxt('GEFCOM.txt')

    actual = np.array([i[2] for i in data[:]])
    predicted = np.array(predict(data))

    print_rate(predicted, actual[360*24:], "ARX - 360-dniowe okno kalibracji")

    plot_prognosis(predicted, actual, 'dla stałego, 360-dniowego okna')

    plot_score(predicted, actual[360*24:], 'ARX z stałym 360-dniowym oknem')
