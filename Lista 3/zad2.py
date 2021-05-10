import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')


if __name__ == "__main__":
    file_name = "GEFCOM.txt"
    x = np.loadtxt(file_name)

    # Zadanie 2a
    plt.scatter(x[:, 2], x[:, 3])
    plt.title("$P_d$$_,$$_h$ vs $Z_d$$_,$$_h$")
    plt.xlabel('$P_d$$_,$$_h$', fontdict={'size': 16})
    plt.ylabel('$Z_d$$_,$$_h$', fontdict={'size': 16})
    plt.show()

    # Zadanie 2b
    for i in range(1, 10):
        plt.scatter(x[:len(x)-i*24:4, 2], x[i*24::4, 2])
        plt.title(f'k = {i}')
        plt.xlabel('$P_d$$_,$$_h$', fontdict={'size': 16})
        plt.ylabel('$P_d$$_-$$_k$$_,$$_h$', fontdict={'size': 16})
        plt.show()
