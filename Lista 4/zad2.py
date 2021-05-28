import numpy as np
import matplotlib.pyplot as plt
from zad1 import predict, set_plot_properties
plt.style.use('ggplot')


if __name__ == "__main__":
    data = np.loadtxt('NPdata_2013-2016.txt')

    actual = data[:, 2]
    score = []
    errors = []
    min_window = 56
    max_window = 361

    for i in range(min_window, max_window):
        print(i)
        predicted = np.array(predict(data, startd=360-i, endd=360, Ndays=776))
        difference = np.array(actual[360 * 24:]) - predicted
        score.append(np.array([np.mean(np.abs(difference)), np.sqrt(np.mean(difference**2))]))
        errors.append(difference)

    np.savetxt('zad2_scores.txt', np.array(score))
    np.savetxt('zad2_errors.txt', np.array(errors))

    y = np.loadtxt('zad2_scores.txt')
    z = np.loadtxt('zad2_errors.txt')
    print(z.shape)

    plt.scatter(np.arange(min_window, max_window, 1), y[:, 0], label='MAE')
    set_plot_properties('MAE w zależności od długości okna kalibracji', 'Długość okna w dniach', 'MAE')
    plt.show()

    plt.scatter(np.arange(min_window, max_window, 1), y[:, 1], label='RMSE')
    set_plot_properties('RMSE w zależności od długości okna kalibracji', 'Długość okna w dniach', 'RMSE')
    plt.show()
