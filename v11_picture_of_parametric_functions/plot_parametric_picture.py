import numpy as np
import matplotlib.pyplot as plt

"""
Тут отрисовывается параметрически заданные функции с помощью Superformula.
Метод plot_param модернизирован, чтобы подписывать графики и сохранять картинки с заданным именем
"""

def r(p, a, b, n1, n2, n3, m, t, name = ""):
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


def plotParam(r, num="", r_name=""):
    """
    Plot parametric function and save to file
    :param r: function
    :param num: number function
    :param r_name: name function
    :return:
    """
    if (num != "" and r_name == ""): # если задали номер функции, но не задали имя
        r_name = name[num]
    title1 = "r" + str(num) + " " + r_name
    title2 = "r" + str(num)

    fig, ax = plt.subplots(figsize=(8, 8))
    t = np.linspace(0 * np.pi, 6 * np.pi, 10000)
    x = r(t) * np.sin(t)
    y = r(t) * np.cos(t)

    #для нормализации фигуры
    #print(max(x))
    #print(max(y))
    #print(max(np.sqrt(x*x+y*y)))

    ax.plot(x, y)
    plt.title(title1)
    plt.savefig(title2 + ".png")
    plt.show()


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


name = [
    'circle',
    'rose',
    'sandglass',
    'modified-square',
    'starfish',
    'bad spiral-flower',
    'modified-triangle blunt corners',
    'modified-triangle sharp corners']


plotParam(r0, num=0)
plotParam(r1, num=1)
plotParam(r2, num=2)
plotParam(r3, num=3)
plotParam(r4, num=4)
plotParam(r5, num=5)
plotParam(r6, num=6)
plotParam(r7, num=7)





