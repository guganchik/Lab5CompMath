import math
import sys

import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable

F = -1

def findClosest(X, x):
    ans = 0
    for i in range(len(X)):
        if x > X[i]:
            ans = i
    return ans
class Solver:
    def __init__(self, X, Y, x):
        self.X = X
        self.Y = Y
        self.x = x
        self.dy = self.getDiff()
        self.closest = findClosest(X, x)
        self.newtonF = self.newton()

    def getDiff(self):
        n = len(self.Y)
        dy = [[self.Y[i]] for i in range(n)]

        for i in range(1, n):
            for j in range(n - i):
                dy[j].append(dy[j + 1][i - 1] - dy[j][i - 1])

        return dy

    def diffTable(self):
        table = PrettyTable()

        fields = ['x_i', 'y_i', '∆y_i']
        for i in range(2, len(self.X)):
            fields.append(f'∆^{i}y_i')

        table.field_names = fields

        dy = [[round(k, 4) for k in i] + [' ' for j in range(len(self.dy) - len(i))] for i in self.dy]

        for i in range(len(self.X)):
            dy[i].insert(0, round(self.X[i], 4))

        table.add_rows(dy)

        return table

    def lagrange(self):
        X = self.X
        x = self.x
        Y = self.Y

        def l(num):
            res = 1

            for j in range(len(X)):
                if j != num:
                    res *= x - X[j]
                    res /= X[num] - X[j]

            return res

        L = 0
        for i in range(len(X)):
            L += Y[i] * l(i)

        return L

    def newtonForward(self, x):
        def T(c):
            resT = 1
            for j in range(c):
                resT *= t - j
            resT /= math.factorial(c)
            return resT

        res = 0
        t = (x - self.X[self.closest]) / (self.X[1] - self.X[0])

        for i in range(len(self.X) - self.closest):
            res += T(i) * self.dy[self.closest][i]

        return res

    def newtonBackward(self, x):
        def T(c):
            res = 1
            for i in range(c):
                res *= t + i
            res /= math.factorial(c)
            return res

        res = 0
        t = (x - self.X[-1]) / (self.X[1] - self.X[0])

        n = len(self.X)
        for i in range(n):
            res += T(i) * self.dy[n - i - 1][i]

        return res

    def newton(self):
        x = self.x
        X = self.X

        if abs(x - X[0]) > abs(x - X[-1]):
            return self.newtonBackward
        return self.newtonForward

    def graph(self):
        X = self.X

        plt.plot(X, self.Y, 'ko')
        x = np.linspace(min(X + [self.x]), max(X + [self.x]), 100)
        plt.plot(x, np.vectorize(self.newtonF)(x))
        plt.show()
def readFromFile():
    name = input('Введите имя файла (или нажмите Enter): ')
    if name == '':
        print('Неверное имя файла')
        return readFromInput()
    else:
        try:
            with open(name) as file:
                X, Y = [], []
                n = int(file.readline().strip())
                for i in range(n):
                    x, y = map(float, file.readline().split())
                    X.append(x)
                    Y.append(y)
                return X, Y
        except:
            print('Произошла ошибка!')
            return readFromInput()


def readFloat():
    try:
        return float(input())
    except ValueError:
        print('Введено не число')
        return readFloat()

def generateTable():
    func = choose_func()

    print('Введите левую границу интервала: ')
    a = readFloat()

    print('Введите правую границу интервала: ')
    b = readFloat()

    if b < a:
        a, b = b, a

    n = input("Введите кол-во точек на интервале (2-15): ")
    while not n.isdigit() or int(n) < 2 or int(n) > 15:
        n = input("Введите корректное кол-во пар (x, y). 2 =< n =< 15. n - целое\n")
    n = int(n)

    step = (b - a) / (n - 1)
    X, Y = [], []
    x = a

    for i in range(n):
        X.append(x)
        Y.append(func(x))

        x += step

    return X, Y


def readRow(file):
    try:
        rows = [float(i) for i in input().strip().split()]
        if len(rows) != 2:
            raise ValueError
        return rows
    except:
        print('Некорректный ввод. Введите: x y пары по одной на строке, через пробел\n')
        if file == 0:
            return readRow(0)
        exit(1)


def readFromInput():
    X, Y = [], []
    n = input("Введите кол-во пар (x, y). 2 =< n =< 15\n")
    while not n.isdigit() or int(n) < 2 or int(n) > 15:
        n = input("Введите корректное кол-во пар (x, y). 2 =< n =< 15. n - целое\n")
    n = int(n)
    for i in range(n):
        x, y = readRow(0)
        X.append(x)
        Y.append(y)
    return X, Y

def getData():
    print('(1) пары значений (x, y) - консоль')
    print('(2) пары значений (x, y) - файл')
    print('(3) На основе выбранной функции')
    input_f = input("Введите необходимый способ (1, 2 или 3) ")
    match input_f:
        case "1":
            return readFromInput()
        case "2":
            return readFromFile()
        case "3":
            return generateTable()
        case _:
            return getData()

def choose_func():
    print("f1 = x^2 + x + 1")
    print("f2 = sin(x)")
    print("f3 = x^3")
    n = input("Выберите функцию:\n")
    while not n.isdigit() or int(n) < 1 or int(n) > len(functions):
        n = input("Выберите функцию (1, 2 или 3):\n")
    n = int(n)
    global FUNC
    FUNC = n - 1
    return functions[n - 1]

def verify(X):
    step = X[1] - X[0]
    if math.isclose(step, 0): return False
    if len(X) == 2: return True

    for i in range(2, len(X)):
        if not math.isclose(X[i] - X[i - 1], step):
            return False

    return True

functions = [
    lambda x: x ** 2 + x + 1,
    lambda x: np.sin(x),
    lambda x: x ** 3
]
def main():
    X, Y = getData()

    if not verify(X):
        print(X)
        print("Введены не равноотстоящие узлы. Выход...")
        exit(1)

    print('Введите значение аргумента:')
    x = readFloat()

    solver = Solver(X, Y, x)

    print(solver.diffTable())
    print('Лагранж:', solver.lagrange())
    print('Ньютон:', solver.newtonF(x))

    if F >= 0:
        print(f'Реальное значение функции: {functions[F](x)}')

    solver.graph()


if __name__ == '__main__':
    main()