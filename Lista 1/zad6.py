import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def set_plot_properties(title, legend_loc, x_label, y_label):
    plt.title(title)
    plt.legend(loc=legend_loc, frameon=False)
    plt.xlabel(x_label, fontdict={'size': 16})
    plt.ylabel(y_label, fontdict={'size': 16})
    plt.show()


x = np.loadtxt("DJIA.txt", comments="#", delimiter="\t", unpack=False)
Z = np.zeros((x.shape[0], 1))

for i in range(1, x.shape[0]):
    Z[i] = np.log(x[i][1] / x[i-1][1])

mu = np.mean(Z)
sigma = np.std(Z)

print('Mu: ' + str(mu))
print('Sigma: ' + str(sigma))

x = np.linspace(-0.1, 0.1, 100)

plt.plot(x, stats.norm.pdf(x, mu, sigma), label='Gęstość prawdopodobieństwa\nz parametrami wyestymowanymi z danych')
plt.hist(Z, bins=30, range=(-0.1, 0.1), density=True, label='Histogram zwrotów')
set_plot_properties('Zwroty logarytmiczne indeksu DJIA', 'upper left', 'x', 'F(x)')


Z = sorted(Z)
y = np.linspace(0, 1, len(Z))
y2 = stats.norm.cdf(Z, mu, sigma)

plt.plot(Z, y, label='Dystrybuanta empiryczna')
plt.plot(Z, y2, label='Dystrybuanta')
set_plot_properties('Skala liniowa', 'upper left', 'x', 'F(x)')

plt.plot(Z, y, label='Dystrybuanta empiryczna')
plt.plot(Z, y2, label='Dystrybuanta')
plt.yscale("log")
set_plot_properties('Skala semi-logarytmiczna', 'upper left', 'x', 'F(x)')

plt.plot(Z, y, label='Dystrybuanta empiryczna')
plt.plot(Z, y2, label='Dystrybuanta')
plt.xscale("log")
plt.yscale("log")
plt.ylim(10**(-4), 10)
set_plot_properties('Skala logarytmiczna', 'upper left', 'x', 'F(x)')
