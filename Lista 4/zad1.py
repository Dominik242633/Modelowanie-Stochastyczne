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
    set_plot_properties(f'Predykcja metodą {title}', 'Czas', 'Wartość')

    plt.plot(np.arange(400*24, 400*24+7*24, 1), actual_data[400*24:400*24+7*24], label='actual')
    plt.plot(np.arange(400*24, 400*24+7*24, 1), prediction[400*24-360*24:400*24+7*24-360*24], label='predicted')
    set_plot_properties(f'Predykcja metodą {title} - w okresie tygodnia', 'Czas', 'Wartość')


def plot_score(prediction, actual, prognosis_name):
    x = np.arange(0, 24, 1)
    mae = [np.mean(np.abs(np.array(prediction[i::24]) - np.array(actual[i::24]))) for i in range(24)]
    rmse = [np.sqrt(np.mean((np.array(prediction[::24]) - np.array(actual[i::24]))**2)) for i in range(24)]

    figure, axis = plt.subplots(2, 1)
    axis[0].bar(x, mae, label='MAE')
    axis[0].set_title(f'MAE dla {prognosis_name}')
    axis[0].set_xlabel('Godzina')
    axis[0].set_ylabel('Wartość')
    axis[0].legend(loc='best', frameon=False)

    axis[1].bar(x, rmse, label='RMSE')
    axis[1].set_title(f'RMSE dla {prognosis_name}')
    axis[1].set_xlabel('Godzina')
    axis[1].set_ylabel('Wartość')
    axis[1].legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.show()


def predict(actual_data, startd=0, endd=360, Ndays=722):
    result = epf_arx(actual_data[:, :4], Ndays, startd, endd, 'arx')[:, 3]
    return result


if __name__ == "__main__":
    data = np.loadtxt('GEFCOM.txt')

    actual = data[:, 2]
    score = []
    errors = []
    min_window = 56
    max_window = 361

    for i in range(min_window, max_window):
        predicted = np.array(predict(data, startd=360-i, endd=360))
        difference = np.array(actual[360 * 24:]) - predicted
        score.append(np.array([np.mean(np.abs(difference)), np.sqrt(np.mean(difference**2))]))
        errors.append(difference)

    np.savetxt('zad1_scores.txt', np.array(score))
    np.savetxt('zad1_errors.txt', np.array(errors))

    y = np.loadtxt('zad1_scores.txt')
    z = np.loadtxt('zad1_errors.txt')

    plt.scatter(np.arange(min_window, max_window, 1), y[:, 0], label='MAE')
    set_plot_properties('MAE w zależności od długości okna kalibracji', 'Długość okna w dniach', 'MAE')
    plt.show()

    plt.scatter(np.arange(min_window, max_window, 1), y[:, 1], label='RMSE')
    set_plot_properties('RMSE w zależności od długości okna kalibracji', 'Długość okna w dniach', 'RMSE')
    plt.show()
