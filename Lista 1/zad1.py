import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
import math


def set_plot_properties(title, legend_loc, x_label, y_label):
    plt.title(title)
    plt.legend(loc=legend_loc, frameon=False)
    plt.xlabel(x_label, fontdict={'size': 16})
    plt.ylabel(y_label, fontdict={'size': 16})
    plt.show()


def plot_hist(numbers, title):
    m = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(-4, 4, 100)

    plt.hist(numbers, density=True, bins=30, label='Gęstość prawdopodobieństwa \nhistogram')
    plt.plot(x, stats.norm.pdf(x, m, sigma), label='Gęstość prawdopodobieństwa')
    set_plot_properties(title, 'upper left', 'x', 'F(x)')


def plot_distribution(numbers, title):
    x = sorted(numbers)
    y = np.linspace(1 / len(x), 1, len(x))

    m = 0
    variance = 1
    sigma = math.sqrt(variance)

    plt.plot(x, y, label='Dystrybuanta empiryczna')
    plt.plot(x, stats.norm.cdf(x, m, sigma), label='Dystrybuanta')
    set_plot_properties(title, 'lower right', 'x', 'F(x)')

    plt.plot(x, y, label='Dystrybuanta empiryczna')
    plt.plot(x, stats.norm.cdf(x, m, sigma), label='Dystrybuanta')
    plt.yscale("log")
    set_plot_properties(title + " - skala semi-logarytmiczna", 'lower right', 'x', 'F(x)')

    plt.plot(x, y, label='Dystrybuanta empiryczna')
    plt.plot(x, stats.norm.cdf(x, m, sigma), label='Dystrybuanta')
    plt.xscale("log")
    plt.yscale("log")
    set_plot_properties(title + " - skala lgarytmiczna", 'lower right', 'x', 'F(x)')


def rule_of_the_dozen(quantity):
    normal_numbers = []
    for i in range(quantity):
        random_numbers = np.array([random.random() for _ in range(12)])
        normal_numbers.append(np.sum(random_numbers, dtype=np.float) - 6)

    return np.array(normal_numbers)


def inverse_transform(quantity):
    uniform_numbers = np.array([random.random() for _ in range(quantity)])
    normal_numbers = stats.norm.ppf(uniform_numbers)

    return normal_numbers


def box_muller(quantity):
    U1 = np.array([random.random() for _ in range(quantity)])
    U2 = np.array([random.random() for _ in range(quantity)])
    normal_numbers = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)

    return normal_numbers


rule_of_the_dozen_numbers = rule_of_the_dozen(1000)
plot_hist(rule_of_the_dozen_numbers, 'Reguła tuzina')
plot_distribution(rule_of_the_dozen(1000), 'Reguła tuzina')

inverse_transform_numbers = inverse_transform(1000)
plot_hist(inverse_transform_numbers, 'Metoda odwróconej dystrybuanty')
plot_distribution(inverse_transform(1000), 'Metoda odwróconej dystrybuanty')

box_muller_numbers = box_muller(1000)
plot_hist(box_muller_numbers, 'Metoda Boxa-Mullera')
plot_distribution(box_muller(1000), 'Metoda Boxa-Mullera')
