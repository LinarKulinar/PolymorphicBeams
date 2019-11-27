import numpy as np
import matplotlib.pyplot as plt
import scipy
import cmath
from scipy.integrate import quad
import functools
import pickle
from datetime import datetime
import time

"""
Тут считается по формуле из Rodrigo 2016 преобразование фурье от параметрических функций.
При отрисовке фокус взят как 1/wavelen, чтобы стандартное ifft из scipy.fftpack по масштабу соответствовало
преобразованию фурье из статьи Rodrigo.
"""


def writedata(size, count, amplitude, phase):
    with open('data_' + str(func_text) + '_num-{:.0f}_size-{:.1f}_pixels-{:.0f}_g-t.pickle'.format(number_pixel_on_mm, size,
                                                                                               count), 'wb') as f:
        pickle.dump(amplitude, f)
        pickle.dump(phase, f)


def loaddata(size, count):
    with open('data_' + str(func_text) + '_num-{:.0f}_size-{:.1f}_pixels-{:.0f}_g-t.pickle'.format(number_pixel_on_mm, size,
                                                                                               count), 'rb') as f:
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


r0 = functools.partial(r, 1, 1, 1, 1, 1, 1, 0)  # circle
r1 = functools.partial(r, 0.41, 1.6, 1, 1.5, 2, 7.5, 12)  # rose
r2 = functools.partial(r, 0.40, 0.9, 10, 4.2, 17, 1.5, 4)  # sandglass (песочные часы)
r3 = functools.partial(r, 0.74, 1, 1, 15, 15, 15, 4)  # modified-square
r4 = functools.partial(r, 0.0001329, 10, 10, 2, 7, 7, 5)  # starfish
r5 = functools.partial(r, 0.81, 1, 1, 5, 5, 5, 10)  # bad spiral, flower
r6 = functools.partial(r, 0.588, 1, 1, 8.5, 15, 15, 3)  # modified-triangle blunt (тупые) corners
r7 = functools.partial(r, 1.04, 0.7, 0.7, 24, 45, 45, 3)  # modified-triangle sharp (острые) corners


def e(x, y):
    """
    :param x: axis x
    :param y: axis y
    :return: amplitude, phase, complex amplitude in the point
    """
    wave_len = 532e-4
    k = 2 * np.pi / wave_len
    f = 1 / wave_len
    g = lambda t: 1  # function dependent on t
    integr_func = lambda t: g(t) * np.exp(
        -1j * k / f * func(t) * (x * np.cos(t) + y * np.sin(t)))  # function under the integral

    field = complex_quadrature(integr_func, 0, 2 * np.pi, limit=200)  # eq.2 from Rodrigo 2018
    return cmath.polar(field[0])  # field  on polar complex


def plot_amplitude(size, count, generate=False):
    # define grid.
    x = np.linspace(-size, size, count+1)  # мы подразумеваем дальше в коде, что поле XY-квадратное
    y = np.linspace(-size, size, count+1)

    plt.subplots(figsize=(8, 8))  # создаем поля для отрисовки
    try:
        if generate:
            raise Exception("Need generate")  # костыль
        z1, z2 = loaddata(size, count)
    except (FileNotFoundError, Exception):
        z1 = np.zeros([count, count])
        z2 = np.zeros([count, count])

        for i in range(count):
            for j in range(count):
                middle = count / 2
                di = abs(i - middle)
                dj = abs(j - middle)
                if di ** 2 + dj ** 2 <= middle ** 2:  # Если мы находимся внутри круга с радиусом, равным count/2
                    z1[i, j], z2[j, i] = e(x[i], y[j])
                    z2[j, i] += cmath.pi
                    print(i, j, '{0:.3f} {0:.3f}'.format(z1[i, j], z2[i, j]))
            # print("column " + str(i) + " is computing")

    writedata(size, count, z1, z2)

    # pcolormesh of interpolated uniform grid with log colormap
    z1_max = np.max(z1)
    z1_min = np.min(z1)
    plt.pcolor(x, y, z1, cmap='gray', vmin=z1_min, vmax=z1_max, label="Амплитуда")
    plt.savefig(str(func_text) + '_num-{:.0f}_size-{:.1f}_pixels-{:.0f}_g-t_AMPLITUDE.png'.format(number_pixel_on_mm, size,
                                                                                              count))
    plt.show()
    plt.subplots(figsize=(8, 8))  # создаем поля для отрисовки
    z2_max = np.max(z2)
    z2_min = np.min(z2)
    plt.pcolor(x, y, z2, cmap='gray', vmin=z2_min, vmax=z2_max, label="Фаза")
    #plt.colorbar()
    plt.savefig(str(func_text) + '_num-{:.0f}_size-{:.1f}_pixels-{:.0f}_g-t_PHASE.png'.format(number_pixel_on_mm, size,
                                                                                          count))
    plt.show()


func = r1  # Отрисовываемая функция
func_text = "r0"
number_pixel_on_mm = 51  # число пикселей на мм
size_image = 1  # размер картинки, которую мы генерируем

print(func_text)
print("number_pixel_on_mm =", number_pixel_on_mm)
print("size_image =", size_image)
summary_pixels_on_field = size_image * number_pixel_on_mm  # Разрешение входной и выходной картинки
summary_pixels_on_field = summary_pixels_on_field if summary_pixels_on_field % 2 == 1 else summary_pixels_on_field + 1  # делаем разрешение не кратным 2
print("summary_pixels_on_field =", summary_pixels_on_field)
print("Вычисления запущены: ", datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S"))
start_time = time.time()  # запоминаем время начала вычислений

plot_amplitude(size_image, summary_pixels_on_field, generate=True)

print("Результат посчитан за: %s секунд" % (time.time() - start_time))
