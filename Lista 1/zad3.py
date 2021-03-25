import numpy as np
import random

# Dokładny wynik całkowania 38/3. Przedziały ufności chyba dobrze, ale ta wariancja coś duża :/

quantity = 1000
a = 1
b = 3
# Rozkład jednostajny na przedziale [a, b]
x = np.array([random.random() for _ in range(quantity)]) * (b - a) + a
y = ((x ** 2 + x) * (b - a))
P = np.sum(y) / len(y)

std_estimator_m_s2 = [np.mean(y), np.var(y)]
CI_std = [np.mean(y) - 1.96 * (np.std(y) / (np.sqrt(quantity))),
          np.mean(y) + 1.96 * (np.std(y) / (np.sqrt(quantity)))]

print('Wartość całki z funkcji: ' + str(P))
print('Wartość dokładna: 38/3, w przybliżeniu: ' + str(38/3))
print('Przedział ufności: ' + str(CI_std))
