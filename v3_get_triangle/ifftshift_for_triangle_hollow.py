import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from numpy.fft import *
import functools
import cmath

"""
тут пытаются отрисовать треугольник с скругленными краями при помоши суперформулы + fftshift
"""


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


def plotParamSubplot(r, number_field_where_plot, text):
    """
    Plot parametric function
    :param r: function
    :return:
    """
    t = np.linspace(0 * np.pi, 2 * np.pi, 10000)
    x = r(t) * np.sin(t)
    y = r(t) * np.cos(t)
    ax[number_field_where_plot].plot(x, y, label="param func")
    ax[number_field_where_plot].set_title(text)


r0 = functools.partial(r, 1, 1, 1, 1, 1, 0)  # circle
r1 = functools.partial(r, 1.6, 1, 1.5, 2, 7.5, 12)  # rose
r2 = functools.partial(r, 0.9, 10, 4.2, 17, 1.5, 4)  # sandglass (песочные часы)
r3 = functools.partial(r, 1, 1, 15, 15, 15, 4)  # modified-square
r4 = functools.partial(r, 10, 10, 2, 7, 7, 5)  # starfish
r5 = functools.partial(r, 1, 1, 5, 5, 5, 10)  # spiral

r6 = functools.partial(r, 1, 1, 8.5, 15, 15, 3)  # modified-triangle blunt (тупые) corners
r7 = functools.partial(r, 0.7, 0.7, 24, 45, 45, 3)  # modified-triangle sharp (острые) corners


def plot_in_gray_subplot(x, y, z, number_field_where_plot, text_on_subplot):
    # pcolormesh of interpolated uniform grid with log colormap
    z_max = np.max(z)
    z_min = np.min(z)
    ax[number_field_where_plot].pcolor(x, y, z, cmap='gray', vmin=z_min, vmax=z_max,
                                       # norm=colors.LogNorm(vmin=z_min, vmax=z_max),
                                       label="Амплитуда")
    ax[number_field_where_plot].set_title(text_on_subplot)


def generate_2D_from_parametric(func, size, count):
    # define grid.
    x = np.linspace(-size, size, count)
    y = np.linspace(-size, size, count)
    #    z = v_e(x, y)

    z0 = np.zeros([len(x), len(y)])

    t = np.linspace(0 * np.pi, 2 * np.pi, 10000)
    xparam = func(t) * np.cos(t)
    yparam = func(t) * np.sin(t)

    for i in range(len(xparam)):
        indx = int(round(xparam[i] / (2 * size / count) + len(x) / 2))
        indy = int(round(yparam[i] / (2 * size / count) + len(y) / 2))
        z0[indx, indy] = 1
        z0[indx - 1, indy] = 1
        z0[indx + 1, indy] = 1
        z0[indx, indy - 1] = 1
        z0[indx, indy + 1] = 1
    return x, y, z0


fig, ax = plt.subplots(2, 2, figsize=(8, 8))  # создаем поля для отрисовки
func = r2  # Отрисовываемая функция
plotParamSubplot(func, (0, 0), "Параметрически")

size_plot_field = 751  # Разрешение входной и выходной картинки
x, y, z = generate_2D_from_parametric(func, 10, size_plot_field)  # генерим растр
plot_in_gray_subplot(x, y, z, (0, 1), "Растровое изображение")

complex_field = ifft2(z, (size_plot_field, size_plot_field
                          ))
complex_field = ifftshift(complex_field)
ampl = np.zeros_like(complex_field, dtype=float)
phase = np.zeros_like(complex_field, dtype=float)

for i in range(len(complex_field)):
    for j in range(len(complex_field[0])):
        ampl[i, j] = cmath.polar(complex_field[i, j])[0]
        phase[i, j] = cmath.polar(complex_field[i, j])[1]
plot_in_gray_subplot(x, y, ampl, (1, 0), "Амплитуда")
plot_in_gray_subplot(x, y, phase, (1, 1), "Фаза")

plt.show()
