import matplotlib.pyplot as plt
from zad1 import load_data
plt.style.use('ggplot')


if __name__ == "__main__":
    file_name = "GEFCOM.txt"
    x = load_data(file_name)

    # Zadanie 2a
    plt.scatter(x[:, 4], x[:, 5])
    plt.title("$P_d$$_,$$_h$ vs $Z_d$$_,$$_h$")
    plt.xlabel('$P_d$$_,$$_h$', fontdict={'size': 16})
    plt.ylabel('$Z_d$$_,$$_h$', fontdict={'size': 16})
    plt.show()

    # Zadanie 2b, nie jest uwzględnione h
    for i in range(1, 10):
        plt.scatter(x[i:, 4], x[:len(x)-i, 4])
        plt.title(f'k = {i}')
        plt.xlabel('$P_d$$_,$$_h$', fontdict={'size': 16})
        plt.ylabel('$P_d$$_-$$_k$$_,$$_h$', fontdict={'size': 16})
        plt.show()
