import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import *
import functools
import cmath
from datetime import datetime
import time

"""
Тут отрисовывается из растра обратное преобразование фурье. В этом файле используются нормированные функции (|r|<=1).
"""


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
    # ax[number_field_where_plot].savefig(
    #    func_text + "_num-" + str(number_pixel_on_mm) + "_size-" + str(size_image) + "_pixels-" + str(
    #        size_plot_field) + "/" + str(number_field_where_plot) + ".png")


r0 = functools.partial(r, 1, 1, 1, 1, 1, 1, 0)  # circle
r1 = functools.partial(r, 0.41, 1.6, 1, 1.5, 2, 7.5, 12)  # rose
r2 = functools.partial(r, 0.40, 0.9, 10, 4.2, 17, 1.5, 4)  # sandglass (песочные часы)
r3 = functools.partial(r, 0.74, 1, 1, 15, 15, 15, 4)  # modified-square
r4 = functools.partial(r, 0.0001329, 10, 10, 2, 7, 7, 5)  # starfish
r5 = functools.partial(r, 0.81, 1, 1, 5, 5, 5, 10)  # bad spiral, flower
r6 = functools.partial(r, 0.588, 1, 1, 8.5, 15, 15, 3)  # modified-triangle blunt (тупые) corners
r7 = functools.partial(r, 1.04, 0.7, 0.7, 24, 45, 45, 3)  # modified-triangle sharp (острые) corners


def plot_in_gray_subplot(x, y, z, number_field_where_plot, text_on_subplot):
    # pcolormesh of interpolated uniform grid with log colormap
    z_max = np.max(z)
    z_min = np.min(z)
    ax[number_field_where_plot].pcolor(x, y, z, cmap='gray', vmin=z_min, vmax=z_max,
                                       # norm=colors.LogNorm(vmin=z_min, vmax=z_max),
                                       label="Амплитуда")
    ax[number_field_where_plot].set_title(text_on_subplot)
    # plt.pcolor(x, y, z, cmap='gray', vmin=z_min, vmax=z_max,
    #            # norm=colors.LogNorm(vmin=z_min, vmax=z_max),
    #            label="Амплитуда")
    # plt.savefig(func_text + "_num-" + str(number_pixel_on_mm) + "_size-" + str(size_image) + "_pixels-" + str(
    #     summary_pixels_on_field) + "/" + str(number_field_where_plot[0]) + str(number_field_where_plot[1]) + ".png")


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
        # z0[indx - 1, indy] = 1
        # z0[indx + 1, indy] = 1
        # z0[indx, indy - 1] = 1
        # z0[indx, indy + 1] = 1
    return x, y, z0


func = r0  # Отрисовываемая функция
func_text = "r0"
number_pixel_on_mm = 25  # число пикселей на мм
size_image = 10  # размер картинки, которую мы генерируем

print(func_text)
print("number_pixel_on_mm =", number_pixel_on_mm)
print("size_image =", size_image)

summary_pixels_on_field = size_image * number_pixel_on_mm  # Разрешение входной и выходной картинки
summary_pixels_on_field = summary_pixels_on_field if summary_pixels_on_field % 2 == 1 else summary_pixels_on_field + 1  # делаем разрешение не кратным 2
print("summary_pixels_on_field =", summary_pixels_on_field)
print("Вычисления запущены: ", datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S"))
start_time = time.time()  # запоминаем время начала вычислений

fig, ax = plt.subplots(2, 2, figsize=(8, 8))  # создаем поля для отрисовки
plotParamSubplot(func, (0, 0), "Параметрически")

x, y, z = generate_2D_from_parametric(func, size_image, summary_pixels_on_field)  # генерим растр
plot_in_gray_subplot(x, y, z, (0, 1), "Растровое изображение")

complex_field = ifft2(z, (summary_pixels_on_field, summary_pixels_on_field))
complex_field = ifftshift(complex_field)
ampl = np.zeros_like(complex_field, dtype=float)
phase = np.zeros_like(complex_field, dtype=float)

for i in range(len(complex_field)):
    for j in range(len(complex_field[0])):
        ampl[i, j] = cmath.polar(complex_field[i, j])[0]
        phase[i, j] = cmath.polar(complex_field[i, j])[1]
plot_in_gray_subplot(x, y, ampl, (1, 0), "Амплитуда")
phase = phase + np.pi
plot_in_gray_subplot(x, y, phase, (1, 1), "Фаза")
plt.savefig(func_text + "_num-" + str(number_pixel_on_mm) + "_size-" + str(size_image) + "_pixels-" + str(
    summary_pixels_on_field) + ".png")
print("Результат посчитан за %s секунд" % (time.time() - start_time))
plt.show()
