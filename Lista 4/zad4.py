import numpy as np
import matplotlib.pyplot as plt
from dmtest import dmtest
from zad1 import set_plot_properties
from scipy.stats import norm
from matplotlib import cm
plt.style.use('ggplot')


def get_avg_score(scores, x):

    mae = np.mean(np.array([scores[i-56, 0] for i in x]))
    rmse = np.mean(np.array([scores[i-56, 1] for i in x]))

    return np.array([mae, rmse])


def plot_heatmap(pvals, title):
    fig, ax = plt.subplots()
    im = ax.imshow(pvals, interpolation='none', vmin=0, vmax=0.2, aspect='equal')
    plt.colorbar(im)
    plt.title(title)
    current_cmap = cm.get_cmap()
    current_cmap.set_bad(color='black')
    ax.set_xticks(np.arange(0, 5, 1))
    ax.set_yticks(np.arange(0, 5, 1))
    ax.set_xticklabels(['AW(56:360)', 'AW(56,360)', 'AW(56,84,112,304,332,360)', 'AW(56)', 'AW(360)'])
    ax.set_yticklabels(['AW(56:360)', 'AW(56,360)', 'AW(56,84,112,304,332,360)', 'AW(56)', 'AW(360)'])
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
    scores = np.loadtxt('zad2_scores.txt')
    errors = np.loadtxt('zad2_errors.txt')

    start_day = 56

    x = ['AW(56:360)',
         'AW(56,360)',
         'AW(56,84,112,304,332,360)',
         'AW(56)',
         'AW(360)']

    y = np.array([get_avg_score(scores, [i for i in range(56, 361)]),
                  get_avg_score(scores, [56, 360]),
                  get_avg_score(scores, [56, 84, 112, 304, 332, 360]),
                  get_avg_score(scores, [56]),
                  get_avg_score(scores, [360])])

    plt.bar(x, y[:, 0], label='MAE')
    set_plot_properties('Uśrednione prognozy MAE', 'Prognoza', 'Wartość')
    plt.show()

    plt.bar(x, y[:, 1], label='RMSE')
    set_plot_properties('Uśrednione prognozy RMSE', 'Prognoza', 'Wartość')
    plt.show()

    prognosis_errors = np.array([np.mean(errors, axis=0),
                                np.mean([errors[56-start_day, :], errors[360-start_day]], axis=0),
                                np.mean([errors[56-start_day, :],
                                         errors[84-start_day, :],
                                         errors[112-start_day, :],
                                         errors[304-start_day, :],
                                         errors[332-start_day, :],
                                         errors[360-start_day]], axis=0),
                                errors[56-start_day, :],
                                errors[360-start_day, :]])

    pvals = np.eye(5)
    r = ['AE', 'SE']

    for lossf in r:
        for index1, error1 in enumerate(prognosis_errors):
            for index2, error2 in enumerate(prognosis_errors):
                if index1 != index2:
                    DM = dmtest(error1, error2, lossf=lossf)
                    pval = 1 - norm.cdf(DM)
                    if pval < 0.2:
                        pvals[index1, index2] = pval
                    else:
                        pvals[index1, index2] = None
                else:
                    pvals[index1, index2] = None

        plot_heatmap(pvals, f"Test Diebolda-Mariano dla r = {1 if lossf == 'AE' else 2}")
