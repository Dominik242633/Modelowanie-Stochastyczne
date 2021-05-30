from christof import christof
import numpy as np
from zad5 import coverage
from zad1 import predict
from zad5 import naive_prediction


if __name__ == '__main__':
    data = np.loadtxt('GEFCOM.txt')

    actual = data[360*24:, 2]

    naive_prognosis = naive_prediction(data)

    naive_coverage50, naive_coverage90 = coverage(actual, naive_prognosis)

    for i in range(24):
        LR_uc_n5, LR_i_n5, LR_cc_n5, LR_uc_p_n5, LR_i_p_n5, LR_cc_p_n5 = christof(naive_coverage50[i::24], 0.5)
        print(f'Naive {i}h for 50% PI - UC={round(LR_uc_p_n5, 2)}, Ind={round(LR_i_p_n5, 2)}, CC={round(LR_cc_p_n5, 2)}')

        LR_uc_n9, LR_i_n9, LR_cc_n9, LR_uc_p_n9, LR_i_p_n9, LR_cc_p_n9 = christof(naive_coverage90[i::24], 0.9)
        print(f'Naive {i}h for 90% PI - UC={round(LR_uc_p_n9, 2)}, Ind={round(LR_i_p_n9, 2)}, CC={round(LR_cc_p_n5, 2)}\n')

    arx_prognosis = np.array(predict(data))

    arx_coverage50, arx_coverage90 = coverage(actual, arx_prognosis)

    for i in range(24):
        LR_uc_n5, LR_i_n5, LR_cc_n5, LR_uc_p_n5, LR_i_p_n5, LR_cc_p_n5 = christof(arx_coverage50[i::24], 0.5)
        print(f'ARX {i}h for 50% PI - UC={round(LR_uc_p_n5, 2)}, Ind={round(LR_i_p_n5, 2)}, CC={round(LR_cc_p_n5, 2)}')

        LR_uc_n9, LR_i_n9, LR_cc_n9, LR_uc_p_n9, LR_i_p_n9, LR_cc_p_n9 = christof(arx_coverage90[i::24], 0.9)
        print(f'ARX {i}h for 90% PI - UC={round(LR_uc_p_n9, 2)}, Ind={round(LR_i_p_n9, 2)}, CC={round(LR_cc_p_n5, 2)}\n')