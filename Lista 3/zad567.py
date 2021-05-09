import numpy as np
import zad5
import zad6
import zad7


if __name__ == "__main__":
    data = np.loadtxt('GEFCOM.txt')

    actual = np.array([i[2] for i in data[:]])

    zad5.print_rate(zad5.predict(data), actual[360*24:], "ARX - 360-dniowe okno kalibracji")

    zad6.print_rate(zad6.predict(data), actual[360*24:], "ARX - rozszerzane okno kalibracji")

    zad7.print_rate(zad7.predict(data), actual[360*24:], "ARX - rolowane okno kalibracji")
