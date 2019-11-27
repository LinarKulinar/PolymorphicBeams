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


def complex_quadrature(func, a, b, **kwargs):
    """
    Calculates the integral of a complex func
    :param func: function
    :param a: left border
    :param b: right border
    :param kwargs:
    :return:
    """

    def real_func(x):
        return scipy.real(func(x))

    def imag_func(x):
        return scipy.imag(func(x))

    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j * imag_integral[0], real_integral[1:], imag_integral[1:])


r0 = functools.partial(r, 1, 1, 1, 1, 1, 0)  # circle
r1 = functools.partial(r, 1.6, 1, 1.5, 2, 7.5, 12)  # rose
r2 = functools.partial(r, 0.9, 10, 4.2, 17, 1.5, 4)  # sandglass (песочные часы)
r3 = functools.partial(r, 1, 1, 15, 15, 15, 4)  # modified-square
r4 = functools.partial(r, 10, 10, 2, 7, 7, 5)  # starfish
r5 = functools.partial(r, 1, 1, 5, 5, 5, 10)  # spiral


def e(x, y):
    """
    :param x: axis x
    :param y: axis y
    :return: amplitude, phase, complex amplitude in the point
    """
    wave_len = 532e-9
    k = 2 * np.pi / wave_len
    f = 0.05
    T = 2 * np.pi
    g = lambda t: t  # function dependent on t
    integr_func = lambda t: g(t) * np.exp(
        -1j * k / f * r1(t) * (x * np.cos(t) + y * np.sin(t)))  # function under the integral

    field = complex_quadrature(integr_func, 0, 2 * np.pi, limit=200)  # eq.2 from Rodrigo 2018
    return cmath.polar(field[0])  # field  on polar complex


def plot_amplitude(size, count, generate=False):
    # define grid.
    x = np.linspace(-size, size, count)
    y = np.linspace(-size, size, count)
    v_e = np.vectorize(e)
    # z = v_e(x, y)

    try:
        if generate:
            raise Exception("Need generate")  # костыль
        z1, z2 = loaddata(size, count)
    except (FileNotFoundError, Exception):
        z1 = np.zeros([len(x), len(y)])
        z2 = np.zeros([len(x), len(y)])

        for i in range(len(x)):
            for j in range(len(y)):
                z1[i, j], z2[j, i] = e(x[i], y[j])
                print(i, j, '{0:.3f} {0:.3f}'.format(z1[i, j], z2[i, j]))

    writedata(size, count, z1, z2)

    # pcolormesh of interpolated uniform grid with log colormap
    z1_max = np.max(z1)
    z1_min = np.min(z1)

    plt.pcolor(x, y, z1, cmap='gray', vmin=z1_min, vmax=z1_max,
               label="Амплитуда")
    plt.show()
    plt.pcolor(x, y, z2, cmap='gray', vmin=0, vmax=2 * np.pi, label="Фаза")
    plt.show()



plot_amplitude(1.2, 25,generate=False)
