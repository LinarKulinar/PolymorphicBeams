import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy
import cmath
from scipy.integrate import quad
import functools
import pickle


def writedata(size, count, amplitude, phase):
    with open('data0_{:.1f}_{:.0f}.pickle'.format(size, count), 'wb') as f:
        pickle.dump(amplitude, f)
        pickle.dump(phase, f)


def loaddata(size, count):
    with open('data0_{:.1f}_{:.0f}.pickle'.format(size, count), 'rb') as f:
        z1 = pickle.load(f)
        z2 = pickle.load(f)
    return z1, z2


def r(p, a, b, n1, n2, n3, m, t):
    """
    :param p: lenth of sweeping
    :param a: option 1
    :param b: option 2
    :param n1: option 3
    :param n2: option 4
    :param n3: option 5
    :param m: option 6
    :param t: polar angle
    :return: nature curve (The Superformula)
    """
    first = np.abs(1 / a * np.cos(m / 4 * t))
    second = np.abs(1 / b * np.sin(m / 4 * t))
    return p * ((first ** n2 + second ** n3) ** (-1 / n1))


def plotParam(r):
    """
    Plot parametric function
    :param r: function
    :return:
    """

    fig, ax = plt.subplots(figsize=(8, 8))  # создаем поля для отрисовки
    t = np.linspace(0 * np.pi, 2 * np.pi, 10000)
    x = r(t) * np.sin(t)
    y = r(t) * np.cos(t)
    ax.plot(x, y, label="param func")
    ax.legend()
    plt.show()


r0 = functools.partial(r, 1, 1, 1, 1, 1, 1, 0)  # circle
r1 = functools.partial(r, 0.41, 1.6, 1, 1.5, 2, 7.5, 12)  # rose
r2 = functools.partial(r, 0.40, 0.9, 10, 4.2, 17, 1.5, 4)  # sandglass (песочные часы)
r3 = functools.partial(r, 0.74, 1, 1, 15, 15, 15, 4)  # modified-square
r4 = functools.partial(r, 0.0001329, 10, 10, 2, 7, 7, 5)  # starfish
r5 = functools.partial(r, 0.81, 1, 1, 5, 5, 5, 10)  # bad spiral, flower

r6 = functools.partial(r, 0.588, 1, 1, 8.5, 15, 15, 3)  # modified-triangle blunt (тупые) corners
r7 = functools.partial(r, 1.04, 0.7, 0.7, 24, 45, 45, 3)  # modified-triangle sharp (острые) corners

plotParam(r0)
plotParam(r1)
plotParam(r2)
plotParam(r3)
plotParam(r4)
plotParam(r5)
plotParam(r6)
plotParam(r7)
