import numpy as np
from christof import christof
from zad1 import predict
from zad5 import naive_prediction, coverage
from datetime import datetime
np.seterr(all="ignore")


if __name__ == "__main__":
    data = np.loadtxt('NPdata_2013-2016.txt')

    weekday = np.array([datetime.strptime(str(int(data[i, 0])), "%Y%m%d").weekday() + 1 for i in range(data.shape[0])])
    weekday = np.reshape(weekday, (weekday.shape[0], 1))
    data = np.hstack((data, weekday))
    naive_prognosis = naive_prediction(data)

    actual = data[360*24:, 2]

    print("Prognoza naiwna\n\t\t50%\t\t90%")
    for i in range(24):
        naive_coverage50, naive_coverage90 = coverage(actual[i::24], naive_prognosis[i::24])

        print(f"Hour {i}: {round(100*np.sum(naive_coverage50)/naive_coverage50.shape[0], 2)}%\t"
              f"{round(100*np.sum(naive_coverage90)/naive_coverage90.shape[0], 2)}%")

    print("\t\t\t\t\t\tNaive prognosis")
    print("\t\t\t\t50%\t\t\t\t\t\t\t90%")
    print("\t\tUC\t\tInd\t\tCC\t\t||\tUC\t\tInd\t\tCC")

    for i in range(24):
        naive_coverage50, naive_coverage90 = coverage(actual[i::24], naive_prognosis[i::24])

        LR_uc_n5, LR_i_n5, LR_cc_n5, LR_uc_p_n5, LR_i_p_n5, LR_cc_p_n5 = christof(naive_coverage50, 0.5)
        LR_uc_n9, LR_i_n9, LR_cc_n9, LR_uc_p_n9, LR_i_p_n9, LR_cc_p_n9 = christof(naive_coverage90, 0.9)

        print(f'Hour {i}\t{LR_uc_p_n5:.2f}\t{LR_i_p_n5:.2f}\t{LR_cc_p_n5:.2f}\t||\t'
              f'{LR_uc_p_n9:.2f}\t{LR_i_p_n9:.2f}\t{LR_cc_p_n9:.2f}')

    arx_prognosis = np.array(predict(data, Ndays=776))

    print("Prognoza ARX\n\t\t50%\t\t90%")
    for i in range(24):
        arx_coverage50, arx_coverage90 = coverage(actual[i::24], arx_prognosis[i::24])

        print(f"Hour {i}: {round(100*np.sum(arx_coverage50)/arx_coverage50.shape[0], 2)}%\t"
              f"{round(100*np.sum(arx_coverage90)/arx_coverage90.shape[0], 2)}%")

    print("\n\t\t\t\t\t\tARX prognosis")
    print("\t\t\t\t50%\t\t\t\t\t\t\t90%")
    print("\t\tUC\t\tInd\t\tCC\t\t||\tUC\t\tInd\t\tCC")

    for i in range(24):
        arx_coverage50, arx_coverage90 = coverage(actual[i::24], arx_prognosis[i::24])

        LR_uc_n5, LR_i_n5, LR_cc_n5, LR_uc_p_n5, LR_i_p_n5, LR_cc_p_n5 = christof(arx_coverage50, 0.5)
        LR_uc_n9, LR_i_n9, LR_cc_n9, LR_uc_p_n9, LR_i_p_n9, LR_cc_p_n9 = christof(arx_coverage90, 0.9)

        print(f'Hour {i}\t{LR_uc_p_n5:.2f}\t{LR_i_p_n5:.2f}\t{LR_cc_p_n5:.2f}\t||\t'
              f'{LR_uc_p_n9:.2f}\t{LR_i_p_n9:.2f}\t{LR_cc_p_n9:.2f}')
