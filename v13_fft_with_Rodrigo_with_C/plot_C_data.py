import numpy as np
import matplotlib.pyplot as plt
import time
import sys
"""
В коде на с++ по формуле из Rodrigo 2016 преобразование фурье от параметрических функций.

В данном файле считывается из файла матрицы и отрисовываются
При отрисовке фокус взят как 1/wavelen, чтобы стандартное ifft из scipy.fftpack по масштабу соответствовало
преобразованию фурье из статьи Rodrigo.
"""


def read_e(path):
    with open(path + "/info.txt", 'rb') as info:
        arr = []
        for line in info:
            arr.append(line)
        param = arr[1][2:-1]
        count = int(arr[2])
        size = float(arr[3])

    with open(path + "/e.txt", 'rb') as f:
        e = np.zeros((count, count))
        for i, line in enumerate(f):
            arr = str(line).split(" ")
            arr = arr[1:-1]
            for j, ampl in enumerate(arr):
                e[i][j] = complex(ampl)
    return e


def read_c_files(path):
    with open(path + "/info.txt", 'rb') as info:
        arr = []
        for line in info:
            arr.append(line)
        param = arr[1][2:-1]
        count = int(arr[2])
        size = float(arr[3])

    with open(path + "/ampl.txt", 'rb') as f:
        amplitude = np.zeros((count, count))
        for i, line in enumerate(f):
            line = line.decode("utf-8")
            arr = str(line).split(" ")[:-1]
            #print(arr)
            for j, ampl in enumerate(arr):
                amplitude[i][j] = float(ampl)

    with open(path + "/intens.txt", 'rb') as f:
        intensive = np.zeros((count, count))
        for i, line in enumerate(f):
            line = line.decode("utf-8")
            arr = str(line).split(" ")[:-1]
            for j, inten in enumerate(arr):
                intensive[i][j] = float(inten)

    with open(path + "/phase.txt", 'rb') as f:
        phase = np.zeros((count, count))
        for i, line in enumerate(f):
            line = line.decode("utf-8")
            arr = str(line).split(" ")[:-1]
            for j, ph in enumerate(arr):
                phase[i][j] = float(ph)

    return param, count, size, amplitude, intensive, phase


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






def plot_data(path):

    param, count, size, amplitude, intensive, phase = read_c_files(path)

    # + np.pi мб надо
    x = np.linspace(-size, size, count + 1)  # мы подразумеваем дальше в коде, что поле XY-квадратное
    y = np.linspace(-size, size, count + 1)

    # Амплитуда
    plt.subplots(figsize=(8, 8))  # создаем поля для отрисовки
    ampl_max = np.max(amplitude)
    ampl_min = np.min(amplitude)
    plt.pcolor(x, y, amplitude, cmap='gray', vmin=ampl_min, vmax=ampl_max,
               label="Амплитуда")
    # plt.colorbar()
    plt.savefig(path+'/_AMPLITUDE.png')
    # plt.show()

    # Интенсивность
    plt.subplots(figsize=(8, 8))  # создаем поля для отрисовки
    intens_max = np.max(intensive)
    intens_min = np.min(intensive)
    plt.pcolor(x, y, intensive, cmap='gray', vmin=intens_min, vmax=intens_max, label="Фаза")
    #plt.colorbar()
    plt.savefig(path+'/_INTENSIVE.png')
    # plt.show()

    # Фаза
    plt.subplots(figsize=(8, 8))  # создаем поля для отрисовки
    phase_max = np.max(phase)
    phase_min = np.min(phase)
    plt.pcolor(x, y, phase, cmap='gray', vmin=phase_min, vmax=phase_max, label="Фаза")
    # plt.colorbar()
    plt.savefig(path + "/PHASE.png")
    #plt.show()


start_time = time.time()  # запоминаем время начала вычислений

path1 = "r0_pixels-21_size-10.000000"
path2 = "r0_pixels-201_size-10.000000"
path3 = "r0_pixels-505_size-5.000000"
path4 = "r0_pixels-505_size-10.000000"
path5 = "r0_pixels-505_size-40.000000"
path6 = "r1_pixels-101_size-5.000000"

print(sys.argv)
params = sys.argv[1:]
if len(params) > 0:
    path = params[0]
    print("path = " + path)
else:
    #path = path1;
    print("Nope param")

plot_data(path)

end_time = time.time()
print("{:g} s".format(end_time - start_time))
