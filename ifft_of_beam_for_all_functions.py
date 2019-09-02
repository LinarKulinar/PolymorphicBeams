import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import *
import functools
import cmath


def r(a, b, n1, n2, n3, m, t):
    """
    :param a: option 1
    :param b: option 2
    :param n1: option 3
    :param n2: option 4
    :param n3: option 5
    :param m: option 6
    :param t: polar angle
    :return: nature curve (The Superformula)
    """
    p = 1  # function dependent on t
    first = np.abs(1 / a * np.cos(m / 4 * t))
    second = np.abs(1 / b * np.sin(m / 4 * t))
    return (first ** n2 + second ** n3) ** (-1 / n1)


def plotParam(r):
    """
    Plot parametric function
    :param r: function
    :return:
    """
    fig, ax = plt.subplots()
    t = np.linspace(0 * np.pi, 2 * np.pi, 10000)
    x = r(t) * np.sin(t)
    y = r(t) * np.cos(t)
    ax.plot(x, y, label="param func")
    ax.legend()
    plt.show()


r0 = functools.partial(r, 1, 1, 1, 1, 1, 0)  # circle
r1 = functools.partial(r, 1.6, 1, 1.5, 2, 7.5, 12)  # rose
r2 = functools.partial(r, 0.9, 10, 4.2, 17, 1.5, 4)  # sandglass (песочные часы)
r3 = functools.partial(r, 1, 1, 15, 15, 15, 4)  # modified-square
r4 = functools.partial(r, 10, 10, 2, 7, 7, 5)  # starfish
r5 = functools.partial(r, 1, 1, 5, 5, 5, 10)  # spiral


def plot_amplitude(x, y, z):
    # pcolormesh of interpolated uniform grid with log colormap
    z_max = np.max(z)
    z_min = np.min(z)
    fig, ax = plt.subplots(figsize=(6, 6))  # Делает соотношение сторон как 1 к 1
    ax.pcolor(x, y, z, cmap='gray', vmin=z_min, vmax=z_max,
              label="Амплитуда")
    plt.show()


def generate_2D_from_parametric(size, count):
    # define grid.
    x = np.linspace(-size, size, count)
    y = np.linspace(-size, size, count)
    #    z = v_e(x, y)

    z0 = np.zeros([len(x), len(y)])

    t = np.linspace(0 * np.pi, 2 * np.pi, 10000)
    xparam = r0(t) * np.sin(t)
    yparam = r0(t) * np.cos(t)

    for i in range(len(xparam)):
        indx = int(round(xparam[i] / (2 * size / count) + len(x) / 2))
        indy = int(round(yparam[i] / (2 * size / count) + len(y) / 2))
        z0[indx, indy] = 1
    return x, y, z0


# plotParam(r0)
x, y, z = generate_2D_from_parametric(1.5, 250)
plot_amplitude(x, y, z)
z1 = ifft2(z, (250, 250))
z2 = np.zeros_like(z1, dtype=float)
z3 = np.zeros_like(z1, dtype=float)

for i in range(len(z1)):
    for j in range(len(z1[0])):
        z2[i, j] = cmath.polar(z1[i, j])[0]
        z3[i, j] = cmath.polar(z1[i, j])[1]
plot_amplitude(x, y, z2)
plot_amplitude(x, y, z3)