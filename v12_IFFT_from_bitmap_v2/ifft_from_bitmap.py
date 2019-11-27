import math

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import *
import cmath
from datetime import datetime
import time
import os

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


def plotParam(r, fig_name="", title="", show_plot = True):  # функция перенесна из v11_picture_of_parametric_functions с сохранением по папочкам
    """
    Plot parametric function and save to file
    :param r: function
    :param fig_name: desciption of generated data
    :param title: decription of graphics
    :return:
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    t = np.linspace(0 * np.pi, 2 * np.pi, 10000)
    x = r(t) * np.sin(t)
    y = r(t) * np.cos(t)
    ax.plot(x, y)
    plt.title(fig_name + "_" + title)
    plt.savefig(fig_name+"/"+title + ".png")
    if (show_plot):
        plt.show()
    else:
        plt.close(fig)


def r0(t):  # circle
    return r(1, 1, 1, 1, 1, 1, 0, t)


def r1(t):  # rose
    return r(0.41, 1.6, 1, 1.5, 2, 7.5, 12, t);


def r2(t):  # sandglass (песочные часы)
    return r(0.40, 0.9, 10, 4.2, 17, 1.5, 4, t);


def r3(t):  # modified-square
    return r(0.74, 1, 1, 15, 15, 15, 4, t);


def r4(t):  # starfish
    return r(0.0001329, 10, 10, 2, 7, 7, 5, t);


def r5(t):  # good spiral
    # Период нужно выставлять на 6pi везде
    return r(0.0438183099 * t, 1, 1, 5, 5, 5, 10, t);


def r6(t):  # modified-triangle blunt (тупые) corners
    return r(0.588, 1, 1, 8.5, 15, 15, 3, t);


def r7(t): # modified-triangle sharp (острые) corners
    return r(1.04, 0.7, 0.7, 24, 45, 45, 3, t);


def plot_in_gray_subplot(x, y, z, fig_name="", title="", show_plot=True):

    fig, ax = plt.subplots(figsize=(8, 8))  # создаем поля для отрисовки
    # pcolormesh of interpolated uniform grid with log colormap
    z_max = np.max(z)
    z_min = np.min(z)
    plt.pcolor(x, y, z, cmap='gray', vmin=z_min, vmax=z_max,
                                       # norm=colors.LogNorm(vmin=z_min, vmax=z_max),
                                       label="Амплитуда")
    plt.title(fig_name + "_" + title)
    plt.savefig(fig_name+"/"+title + ".png") # в папочку положили
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def generate_2D_from_parametric(func, size, count):
    # define grid.
    x = np.linspace(-size, size, count)
    y = np.linspace(-size, size, count)

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


name = [
    'circle',
    'rose',
    'sandglass',
    'modified-square',
    'starfish',
    'bad spiral-flower',
    'modified-triangle blunt corners',
    'modified-triangle sharp corners']



# Вводим данные для расчетов:
func = r5  # Отрисовываемая функция
num_func = 5  # номер функции
show_plot = False
abs_tol = 1e-10
number_pixel_on_mm = 101  # число пикселей на мм
size_image = 5  # размер картинки, которую мы генерируем



func_text = "r" + str(num_func)
summary_pixels_on_field = size_image * number_pixel_on_mm  # Разрешение входной и выходной картинки
summary_pixels_on_field = summary_pixels_on_field if summary_pixels_on_field % 2 == 1 else summary_pixels_on_field + 1  # делаем разрешение не кратным 2

print(func_text)
print("name =", name[num_func])
print("number_pixel_on_mm =", number_pixel_on_mm)
print("size_image =", size_image)
print("summary_pixels_on_field =", summary_pixels_on_field)
print("Вычисления запущены: ", datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S"))

decription_data = func_text + "_num-" +str(number_pixel_on_mm) + "_size-"+ str(size_image) + "_pixels-" + str(summary_pixels_on_field) #так будем называть папку и данные

if not os.path.exists(decription_data):
    os.makedirs(decription_data)  # создаем папку, которая называется, как данный набор параметров

plotParam(func, fig_name=decription_data, title="1Параметрически", show_plot=show_plot)

start_time = time.time()  # запоминаем время начала вычислений

x, y, z = generate_2D_from_parametric(func, size_image, summary_pixels_on_field)  # генерим растр
plot_in_gray_subplot(x, y, z, fig_name=decription_data, title="2Растровое изображение", show_plot=show_plot)

complex_field = ifft2(z, (summary_pixels_on_field, summary_pixels_on_field))
complex_field = ifftshift(complex_field)
ampl = np.zeros_like(complex_field, dtype=float)  # мгновенная напряженность
intensive = np.zeros_like(complex_field, dtype=float)  # квадрат амплитуды
phase = np.zeros_like(complex_field, dtype=float)  # значение фазы

time_computing = time.time()
print("БПФ посчитан за %s секунд" % (time_computing - start_time))

for i in range(len(complex_field)):
    for j in range(len(complex_field[0])):
        middle = len(complex_field) / 2
        di = abs(i - middle)
        dj = abs(j - middle)
        if di ** 2 + dj ** 2 <= middle ** 2:
            # стоит обратить внимание, что в данных фурье ещё есть значения
            # за пределами круга радиусом len(complex_field) / 2
            ampl[i, j] = cmath.polar(complex_field[i, j])[0]
            phase[i, j] = cmath.polar(complex_field[i, j])[1] + np.pi
            if math.isclose(phase[i, j], 2*np.pi, abs_tol=abs_tol):  # cast 2pi -> 0
                phase[i, j] = phase[i, j]-2*np.pi
            intensive[i, j] = ampl[i, j]**2
plot_in_gray_subplot(x, y, ampl, fig_name=decription_data, title="Амплитуда", show_plot=show_plot)
plot_in_gray_subplot(x, y, intensive, fig_name=decription_data, title="Интенсивность", show_plot=show_plot)
plot_in_gray_subplot(x, y, phase, fig_name=decription_data, title="Фаза_tol="+str(abs_tol), show_plot=show_plot)

time_complex = time.time()
print("Преобразования над комплексными числами посчитаны за %s секунд" % (time_complex-time_computing))

# Пишем info в файл
info = open(decription_data + "/" + "info.txt", 'w')
info.write(func_text + "\n")
info.write("name = " + str(name[num_func]) + "\n")
info.write("number_pixel_on_mm = " + str(number_pixel_on_mm) + "\n")
info.write("size_image = " + str(size_image) + "\n")
info.write("summary_pixels_on_field = " + str(summary_pixels_on_field) + "\n")
info.write("Вычисления запущены: " + str(datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")) + "\n")
info.write("БПФ посчитан за %s секунд " % (time_computing - start_time)+ "\n")
info.write("Преобразования над комплексными числами посчитаны за %s секунд" % (time_complex-time_computing))
info.close()

plt.show()
