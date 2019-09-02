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


def f(x, sigma=1):
    return np.exp(-(x ** 2 / (sigma ** 2)))


def f_fourier(u):
    integr_func = lambda t: 1/np.sqrt(2*np.pi)*f(t) * np.exp(-1j * 2 * np.pi * t * u)  # function under the integral
    result_integrate = complex_quadrature(integr_func, -np.Inf, np.Inf)
    return cmath.polar(result_integrate[0])[0]


# v_f_fourier = np.vectorize(f_fourier)
# def e(x, y):
#     """
#     :param x: axis x
#     :param y: axis y
#     :return: amplitude, phase, complex amplitude in the point
#     """
#     wave_len = 532e-9
#     k = 2 * np.pi / wave_len
#     f = 0.05
#     T = 2 * np.pi
#     g = lambda t: 1  # function dependent on t
#     integr_func = lambda t: g(t) * np.exp(
#         -1j * k / f * r1(t) * (x * np.cos(t) + y * np.sin(t)))  # function under the integral
#
#     field = complex_quadrature(integr_func, 0, 2 * np.pi, limit=200)  # eq.2 from Rodrigo 2018
#     return cmath.polar(field[0])  # field  on polar complex
#
#

def plot_2d(f, size, count):
    t = np.linspace(-size, size, count)
    func = np.zeros(t.shape)
    for idx in range(t.shape[0]):
        func[idx] = f(t[idx])
    plt.plot(t, func)
    plt.show()


plot_2d(f, 10, 300)
plot_2d(f_fourier, 10, 300)
