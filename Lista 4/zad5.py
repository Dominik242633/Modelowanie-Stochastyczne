import numpy as np
from epf_arx import epf_arx


def predict(actual_data, startd=0, endd=360, Ndays=722):
    result = epf_arx(actual_data[:, :4], Ndays, startd, endd, 'arx')[:, 3]
    return result


def naive_prediction(actual):
    prediction = []
    for i in range(360 * 24, actual.shape[0]):
        if actual[i, -1] in [1, 6, 7]:
            prediction.append(actual[i - 7 * 24, 2])
        else:
            prediction.append(actual[i - 24, 2])
    return prediction


def coverage(actual, predicted):
    PI50 = np.zeros((actual.shape[0], 2))
    cover50 = np.zeros(actual.shape[0])

    PI90 = np.zeros((actual.shape[0], 2))
    cover90 = np.zeros(actual.shape[0])

    for j in range(361, len(actual)):
        error = actual[j - 361:j] - predicted[j - 361:j]
        PI50[j, 0] = predicted[j] + np.percentile(error, 25)
        PI50[j, 1] = predicted[j] + np.percentile(error, 75)
        cover50[j] = (actual[j] > PI50[j, 0]) and (actual[j] < PI50[j, 1])

        PI90[j, 0] = predicted[j] + np.percentile(error, 5)
        PI90[j, 1] = predicted[j] + np.percentile(error, 95)
        cover90[j] = (actual[j] > PI90[j, 0]) and (actual[j] < PI90[j, 1])

    return [cover50[361:], cover90[361:]]


if __name__ == "__main__":
    data = np.loadtxt('GEFCOM.txt')

    actual = data[360 * 24:, 2]

    naive_prognosis = naive_prediction(data)
    print("Prognoza naiwna\n\t\t50%\t\t90%")
    for i in range(24):
        naive_coverage50, naive_coverage90 = coverage(actual[i::24], naive_prognosis[i::24])

        print(f"Hour {i}: {round(100 * np.sum(naive_coverage50) / naive_coverage50.shape[0], 2)}%\t"
              f"{round(100 * np.sum(naive_coverage90) / naive_coverage90.shape[0], 2)}%")

    arx_prognosis = np.array(predict(data))

    print("Prognoza ARX\n\t\t50%\t\t90%")
    for i in range(24):
        arx_coverage50, arx_coverage90 = coverage(actual[i::24], arx_prognosis[i::24])

        print(f"Hour {i}: {round(100 * np.sum(arx_coverage50) / arx_coverage50.shape[0], 2)}%\t"
              f"{round(100 * np.sum(arx_coverage90) / arx_coverage90.shape[0], 2)}%")
