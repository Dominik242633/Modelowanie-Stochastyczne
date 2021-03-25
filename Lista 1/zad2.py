import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats as stats


def set_plot_properties(title, legend_loc, x_label, y_label):
    plt.title(title)
    plt.legend(loc=legend_loc, frameon=False)
    plt.xlabel(x_label, fontdict={'size': 16})
    plt.ylabel(y_label, fontdict={'size': 16})
    plt.show()


def plot_hist(numbers, title, type):

    if type == 'lognormal':
        x = np.linspace(0, max(numbers), 100)
        plt.plot(x, stats.lognorm.pdf(x, 1/4), label='Gęstość prawdopodobieństwa')
    elif type == 'pareto':
        x = np.linspace(0, max(numbers), 100)
        plt.plot(x, stats.pareto.pdf(x, 3), label='Gęstość prawdopodobieństwa')
    elif type == 'exponential':
        x = np.linspace(0, 4, 100)
        plt.plot(x, stats.expon.pdf(x), label='Gęstość prawdopodobieństwa')
    else:
        print('Wrong type !')
        return False

    plt.hist(numbers, density=True, bins=30, label='Gęstość prawdopodobieństwa \nhistogram')
    set_plot_properties(title, 'upper right', 'x', 'F(x)')


def plot_distribution(numbers, title, type):
    x = sorted(numbers)
    y = np.linspace(1 / len(x), 1, len(x))

    if type == 'lognormal':
        y2 = stats.lognorm.cdf(x, 1/4)
    elif type == 'pareto':
        y2 = stats.pareto.cdf(x, 3)
    elif type == 'exponential':
        y2 = stats.expon.cdf(x)
    else:
        print('Wrong type !')
        return False

    plt.plot(x, y, label='Dystrybuanta empiryczna')
    plt.plot(x, y2, label='Dystrybuanta')
    set_plot_properties(title, 'lower right', 'x', 'F(x)')

    plt.plot(x, y, label='Dystrybuanta empiryczna')
    plt.plot(x, y2, label='Dystrybuanta')
    plt.yscale("log")
    set_plot_properties(title + " - skala semi-logarytmiczna", 'lower right', 'x', 'F(x)')

    plt.plot(x, y, label='Dystrybuanta empiryczna')
    plt.plot(x, y2, label='Dystrybuanta')
    plt.xscale("log")
    plt.yscale("log")
    set_plot_properties(title + " - skala logarytmiczna", 'lower right', 'x', 'F(x)')


def lognormal_distribution(quantity):
    '''Funkcja zwraca liczby z rozkładu lognormalnego'''

    X = np.exp(np.random.normal(0, 1/4, quantity))

    return X


def pareto_distribution(quantity):
    '''Funkcja zwraca liczby z rozkładu pareto, generowane
    metodą odwróconej dystrybuanty'''
    lamb = 1
    alpha = 3

    U = np.array([random.random() for _ in range(quantity)])
    X = lamb * (U ** (-1 / alpha))

    return X


def exponential_distribution(quantity):
    '''Funkcja zwraca liczby z rozkładu wykładniczego, generowane
    metodą odwróconej dystrybuanty'''

    beta = 1
    U = np.array([random.random() for _ in range(quantity)])
    X = (-1 / beta) * np.log(U)

    return X


lognormal_distribution_numbers = lognormal_distribution(1000)
plot_hist(lognormal_distribution_numbers, 'Rozkład lognormalny', 'lognormal')
plot_distribution(lognormal_distribution_numbers, 'Rozkład lognormalny', 'lognormal')

pareto_distribution_numbers = pareto_distribution(1000)
plot_hist(pareto_distribution_numbers, 'Rozkład pareto', 'pareto')
plot_distribution(pareto_distribution_numbers, 'Rozkład pareto', 'pareto')

exponential_distribution_numbers = exponential_distribution(1000)
plot_hist(exponential_distribution_numbers, 'Rozkład wykładniczy', 'exponential')
plot_distribution(exponential_distribution_numbers, 'Rozkład wykładniczy', 'exponential')
