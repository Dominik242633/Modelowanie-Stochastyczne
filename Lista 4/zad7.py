import numpy as np
from christof import christof
from zad1 import predict
from zad5 import naive_prediction, coverage
from datetime import datetime


if __name__ == "__main__":
    data = np.loadtxt('NPdata_2013-2016.txt')

    weekday = np.array([datetime.strptime(str(int(data[i, 0])), "%Y%m%d").weekday() + 1 for i in range(data.shape[0])])
    weekday = np.reshape(weekday, (weekday.shape[0], 1))
    data = np.hstack((data, weekday))
    naive_prognosis = naive_prediction(data)

    actual = data[360*24:, 2]

    naive_coverage50, naive_coverage90 = coverage(actual, naive_prognosis)

    print("Prognoza naiwna")
    print(f"50% przedział - pokrycie: {round(100*np.sum(naive_coverage50)/actual.shape[0], 2)}%")
    print(f"90% przedział - pokrycie: {round(100*np.sum(naive_coverage90)/actual.shape[0], 2)}%")

    for i in range(24):
        LR_uc_n5, LR_i_n5, LR_cc_n5, LR_uc_p_n5, LR_i_p_n5, LR_cc_p_n5 = christof(naive_coverage50[i::24], 0.5)
        print(f'Naive {i}h for 50% PI - UC={round(LR_uc_p_n5, 2)}, Ind={round(LR_i_p_n5, 2)}, CC={round(LR_cc_p_n5, 2)}')

        LR_uc_n9, LR_i_n9, LR_cc_n9, LR_uc_p_n9, LR_i_p_n9, LR_cc_p_n9 = christof(naive_coverage90[i::24], 0.9)
        print(f'Naive {i}h for 90% PI - UC={round(LR_uc_p_n9, 2)}, Ind={round(LR_i_p_n9, 2)}, CC={round(LR_cc_p_n5, 2)}\n')

    arx_prognosis = np.array(predict(data, Ndays=776))

    arx_coverage50, arx_coverage90 = coverage(actual, arx_prognosis)

    print("Prognoza ARX")
    print(f"50% przedział - pokrycie: {round(100*np.sum(arx_coverage50)/actual.shape[0], 2)}%")
    print(f"90% przedział - pokrycie: {round(100*np.sum(arx_coverage90)/actual.shape[0], 2)}%")

    for i in range(24):
        LR_uc_n5, LR_i_n5, LR_cc_n5, LR_uc_p_n5, LR_i_p_n5, LR_cc_p_n5 = christof(arx_coverage50[i::24], 0.5)
        print(f'ARX {i}h for 50% PI - UC={round(LR_uc_p_n5, 2)}, Ind={round(LR_i_p_n5, 2)}, CC={round(LR_cc_p_n5, 2)}')

        LR_uc_n9, LR_i_n9, LR_cc_n9, LR_uc_p_n9, LR_i_p_n9, LR_cc_p_n9 = christof(arx_coverage90[i::24], 0.9)
        print(f'ARX {i}h for 90% PI - UC={round(LR_uc_p_n9, 2)}, Ind={round(LR_i_p_n9, 2)}, CC={round(LR_cc_p_n5, 2)}\n')
