from christof import christof
import numpy as np
from zad5 import coverage, naive_prediction
from zad1 import predict


if __name__ == '__main__':
    data = np.loadtxt('GEFCOM.txt')

    actual = data[360*24:, 2]

    naive_prognosis = naive_prediction(data)

    print("\t\t\t\t\t\tNaive prognosis")
    print("\t\t\t\t50%\t\t\t\t\t\t\t90%")
    print("\t\tUC\t\tInd\t\tCC\t\t||\tUC\t\tInd\t\tCC")
    for i in range(24):
        naive_coverage50, naive_coverage90 = coverage(actual[i::24], naive_prognosis[i::24])

        LR_uc_n5, LR_i_n5, LR_cc_n5, LR_uc_p_n5, LR_i_p_n5, LR_cc_p_n5 = christof(naive_coverage50, 0.5)
        LR_uc_n9, LR_i_n9, LR_cc_n9, LR_uc_p_n9, LR_i_p_n9, LR_cc_p_n9 = christof(naive_coverage90, 0.9)

        print(f'Hour {i}\t{LR_uc_p_n5:.2f}\t{LR_i_p_n5:.2f}\t{LR_cc_p_n5:.2f}\t||\t'
              f'{LR_uc_p_n9:.2f}\t{LR_i_p_n9:.2f}\t{LR_cc_p_n9:.2f}')

    arx_prognosis = np.array(predict(data))

    print("\n\t\t\t\t\t\tARX prognosis")
    print("\t\t\t\t50%\t\t\t\t\t\t\t90%")
    print("\t\tUC\t\tInd\t\tCC\t\t||\tUC\t\tInd\t\tCC")

    for i in range(24):
        arx_coverage50, arx_coverage90 = coverage(actual[i::24], arx_prognosis[i::24])

        LR_uc_n5, LR_i_n5, LR_cc_n5, LR_uc_p_n5, LR_i_p_n5, LR_cc_p_n5 = christof(arx_coverage50, 0.5)
        LR_uc_n9, LR_i_n9, LR_cc_n9, LR_uc_p_n9, LR_i_p_n9, LR_cc_p_n9 = christof(arx_coverage90, 0.9)

        print(f'Hour {i}\t{LR_uc_p_n5:.2f}\t{LR_i_p_n5:.2f}\t{LR_cc_p_n5:.2f}\t||\t'
              f'{LR_uc_p_n9:.2f}\t{LR_i_p_n9:.2f}\t{LR_cc_p_n9:.2f}')
