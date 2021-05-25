import numpy as np
from dmtest import dmtest
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import cm
import zad4
import zad5
import zad6
import zad7


def get_prognosis():
    data = np.loadtxt('GEFCOM.txt')

    actual = data[360*24:, 2]

    naive_method_err = actual - np.array(zad4.naive_prediction(data))

    hw_prediction = np.zeros(data.shape)
    for hour in range(24):
        hw_prediction[360*24+hour::24, 2] = zad4.hw_method(data[hour::24, 2])
    hw_method_err = actual - hw_prediction[360*24:, 2]

    const_arx_err = actual - np.array(zad5.predict(data))
    expanded_arx_err = actual - np.array(zad6.predict(data))
    rolled_arx_err = actual - np.array(zad7.predict(data))

    errors = [naive_method_err,
              hw_method_err,
              const_arx_err,
              expanded_arx_err,
              rolled_arx_err]

    return errors


def plot_heatmap(pvals, title):
    fig, ax = plt.subplots()
    im = ax.imshow(pvals, interpolation='none', vmin=0, vmax=1, aspect='equal')
    plt.colorbar(im)
    plt.title(title)
    current_cmap = cm.get_cmap()
    current_cmap.set_bad(color='black')
    ax.set_xticks(np.arange(0, 5, 1))
    ax.set_yticks(np.arange(0, 5, 1))
    ax.set_xticklabels(['Naive', 'HW', 'Constant ARX', 'Expanded ARX', 'Rolled ARX'])
    ax.set_yticklabels(['Naive', 'HW', 'Constant ARX', 'Expanded ARX', 'Rolled ARX'])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.grid(None)
    for i in range(pvals.shape[0]):
        for j in range(pvals.shape[1]):
            if not np.isnan(pvals[i, j]):
                text = ax.text(j, i, round(pvals[i, j], 3), ha="center", va="center", color="w")
            if i == j:
                text = ax.text(j, i, 'X', ha="center", va="center", color="w", fontsize=20.0)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":

    errors_hour = get_prognosis()
    errors_day = errors_hour.copy()

    errors_hour = [prognosis_error[8::24] for prognosis_error in errors_hour]
    errors_day = [[np.mean(prognosis_error[i*24:i*24+24]) for i in range(722)] for prognosis_error in errors_day]

    pvals = np.eye(5)

    for index1, error1 in enumerate(errors_hour):
        for index2, error2 in enumerate(errors_hour):
            if index1 != index2:
                DM = dmtest(error1, error2, lossf='SE')
                pval = 1 - norm.cdf(DM)
                if pval < 0.2:
                    pvals[index1, index2] = 1 - norm.cdf(DM)
                else:
                    pvals[index1, index2] = None
            else:
                pvals[index1, index2] = None

    plot_heatmap(pvals, "Test Diebolda-Mariano dla godziny 8")

    for index1, error1 in enumerate(errors_day):
        for index2, error2 in enumerate(errors_day):
            if index1 != index2:
                DM = dmtest(error1, error2, lossf='AE')
                pval = 1 - norm.cdf(DM)
                if pval < 0.2:
                    pvals[index1, index2] = pval
                else:
                    pvals[index1, index2] = None

    plot_heatmap(pvals, "Test Diebolda-Mariano dla średnich dziennych")
