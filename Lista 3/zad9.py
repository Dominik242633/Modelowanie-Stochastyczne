import numpy as np
from dmtest import dmtest
from scipy.stats import norm
import zad5
import zad6

if __name__ == "__main__":
    data = np.loadtxt('GEFCOM.txt')

    hour = 3

    actual = np.array([i[2] for i in data[:]])[hour::24]

    zad5_err_predicted = actual[360:] - np.array(zad5.predict(data))[hour::24]
    zad6_err_predicted = actual[360:] - np.array(zad6.predict(data))[hour::24]

    DM = dmtest(zad5_err_predicted, zad6_err_predicted, 1, 'AE')
    DM1_pval = 1 - norm.cdf(DM)
    DM = dmtest(zad5_err_predicted, zad6_err_predicted, 1, 'SE')
    DM2_pval = 1 - norm.cdf(DM)

    print(f'Diebold-Mariano test for hour {hour}, HW significantly better if p-value <0.05')
    print('\t\tAE\t\t\t\t\tSE')
    print(DM1_pval, DM2_pval)
