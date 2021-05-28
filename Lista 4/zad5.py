import numpy as np
import matplotlib.pyplot as plt
from zad1 import predict


def naive_prediction(actual):
    prediction = []
    for i in range(360*24, actual.shape[0]):
        if actual[i, -1] in [1, 6, 7]:
            prediction.append(actual[i-7*24, 2])
        else:
            prediction.append(actual[i-24, 2])
    return prediction


def coverage(actual, predicted):
    PI50 = np.zeros((actual.shape[0], 2))
    PI90 = np.zeros((actual.shape[0], 2))

    naive_error = actual - predicted

    PI50[:, 0] = naive_prognosis + np.percentile(naive_error, 25)
    PI50[:, 1] = naive_prognosis + np.percentile(naive_error, 75)
    cover50 = np.array(
        [(actual[index] > element[0]) and (actual[index] < element[1]) for index, element in enumerate(PI50)])

    PI90[:, 0] = naive_prognosis + np.percentile(naive_error, 5)
    PI90[:, 1] = naive_prognosis + np.percentile(naive_error, 95)
    cover90 = np.array(
        [(actual[index] > element[0]) and (actual[index] < element[1]) for index, element in enumerate(PI90)])

    return [np.sum(cover50), np.sum(cover90)]


if __name__ == "__main__":
    data = np.loadtxt('GEFCOM.txt')

    actual = data[360*24:, 2]

    naive_prognosis = naive_prediction(data)

    naive_coverage50, naive_coverage90 = coverage(actual, naive_prognosis)

    print("Prognoza naiwna")
    print(np.sum(naive_coverage50), np.sum(naive_coverage90))
    print(f"50% przedział - pokrycie: {round(100*np.sum(naive_coverage50)/actual.shape[0], 2)}%")
    print(f"90% przedział - pokrycie: {round(100*np.sum(naive_coverage90)/actual.shape[0], 2)}%")

    arx_prognosis = np.array(predict(data))

    arx_coverage50, arx_coverage90 = coverage(actual, arx_prognosis)

    print("Prognoza ARX")
    print(f"50% przedział - pokrycie: {round(100*np.sum(arx_coverage50)/actual.shape[0], 2)}%")
    print(f"90% przedział - pokrycie: {round(100*np.sum(arx_coverage90)/actual.shape[0], 2)}%")
