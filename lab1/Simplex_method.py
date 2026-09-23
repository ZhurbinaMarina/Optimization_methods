import numpy as np
from fractions import Fraction


class SimplexSolver:
    def __init__(self, objective, constraints, b, signs, maximize=True):
        self.orig_obj = [Fraction(x) for x in objective]
        self.maximize = maximize
        self.constraints = [[Fraction(x) for x in row] for row in constraints]
        self.b = [Fraction(x) for x in b]
        self.signs = signs

        self.num_orig_vars = len(objective)
        self.num_constraints = len(b)

        self.tableau = None
        self.basis = []
        self.artificial_vars = []
        self.var_names = []

    def print_tableau(self, step_name):
        print(f"\n--- {step_name} ---")
        header = "Базис | " + " | ".join(f"{name:>5}" for name in self.var_names) + " |     b"
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        for i in range(self.num_constraints):
            bas_name = self.var_names[self.basis[i]]
            row_str = " | ".join(f"{str(x):>5}" for x in self.tableau[i, :-1])
            print(f"{bas_name:>3} | {row_str} | {str(self.tableau[i, -1]):>5}")

        print("-" * len(header))
        delta_str = " | ".join(f"{str(x):>5}" for x in self.tableau[-1, :-1])
        print(f"  Δ | {delta_str} | {str(self.tableau[-1, -1]):>5}")
        print("-" * len(header))

    def setup(self):
        # Приведение к каноническому виду
        if self.maximize:
            self.c = [-x for x in self.orig_obj]
        else:
            self.c = list(self.orig_obj)

        self.var_names = [f"x{i + 1}" for i in range(self.num_orig_vars)]

        # Добавляем дополнительные и искусственные переменные
        slacks = []
        artificials = []

        for i in range(self.num_constraints):
            if self.b[i] < 0:
                self.b[i] = -self.b[i]
                self.constraints[i] = [-x for x in self.constraints[i]]
                self.signs[i] = '>=' if self.signs[i] == '<=' else '<=' if self.signs[i] == '>=' else '='

            if self.signs[i] == '<=':
                slacks.append((i, 1))
                self.basis.append(len(self.var_names))
                self.var_names.append(f"x{len(self.var_names) + 1}")
            elif self.signs[i] == '>=':
                slacks.append((i, -1))
                self.var_names.append(f"x{len(self.var_names) + 1}")

                artificials.append(i)
                self.basis.append(len(self.var_names))
                self.artificial_vars.append(len(self.var_names))
                self.var_names.append(f"x{len(self.var_names) + 1}")
            elif self.signs[i] == '=':
                artificials.append(i)
                self.basis.append(len(self.var_names))
                self.artificial_vars.append(len(self.var_names))
                self.var_names.append(f"x{len(self.var_names) + 1}")

        # Формируем матрицу
        num_vars = len(self.var_names)
        self.tableau = np.zeros((self.num_constraints + 1, num_vars + 1), dtype=object)

        # Заполняем основные переменные
        for i in range(self.num_constraints):
            self.tableau[i, :self.num_orig_vars] = self.constraints[i]
            self.tableau[i, -1] = self.b[i]

        # Заполняем дополнительные и искусственные переменные
        for idx, (row, val) in enumerate(slacks):
            col = self.num_orig_vars + idx
            self.tableau[row, col] = Fraction(val)

        for idx, row in enumerate(artificials):
            col = self.num_orig_vars + len(slacks) + idx
            self.tableau[row, col] = Fraction(1)

    def pivot(self, pivot_row, pivot_col):
        # Пересчитываем таблциу
        pivot_val = self.tableau[pivot_row, pivot_col]
        self.tableau[pivot_row, :] /= pivot_val

        for i in range(len(self.tableau)):
            if i != pivot_row:
                self.tableau[i, :] -= self.tableau[i, pivot_col] * self.tableau[pivot_row, :]

        self.basis[pivot_row] = pivot_col

    def solve_phase(self, phase_name):
        iteration = 1
        while True:
            self.print_tableau(f"{phase_name} - Итерация {iteration}")
            delta_row = self.tableau[-1, :-1]

            # Ищем разрешающий столбец
            min_val = min(delta_row)
            if min_val >= 0:
                print("Отрицательных элементов в Δ нет. Фаза завершена.")
                break

            pivot_col = np.argmin(delta_row)

            # Ищем разрешающую строку
            ratios = []
            for i in range(self.num_constraints):
                val = self.tableau[i, pivot_col]
                if val > 0:
                    ratios.append(self.tableau[i, -1] / val)
                else:
                    ratios.append(float('inf'))

            min_ratio = min(ratios)
            if min_ratio == float('inf'):
                raise Exception("Решение не ограничено (целевая функция уходит в бесконечность)")

            pivot_row = ratios.index(min_ratio)
            print(f"-> Входит в базис: переменная {self.var_names[pivot_col]}, Выходит: переменная {self.var_names[self.basis[pivot_row]]}")
            print(f"-> Разрешающий элемент: {self.tableau[pivot_row, pivot_col]}")

            # Пересчет таблицы
            self.pivot(pivot_row, pivot_col)
            iteration += 1

    def solve(self):
        self.setup()

        # Вспомогательная задача
        if self.artificial_vars:
            print("\nРешение вспомогательной задачи:")
            # Целевая функция: сумма искусственных переменных
            for art_col in self.artificial_vars:
                self.tableau[-1, art_col] = Fraction(1)

            # Подстановка
            for i in range(self.num_constraints):
                bas_col = self.basis[i]
                if self.tableau[-1, bas_col] != 0:
                    self.tableau[-1, :] -= self.tableau[i, :] * self.tableau[-1, bas_col]

            self.solve_phase("Фаза 1")

            if self.tableau[-1, -1] < 0:  # С учетом знака Q
                raise Exception("Система ограничений несовместна (пустое множество решений)")

            # Удаляем столбцы искусственных переменных
            cols_to_keep = [i for i in range(len(self.var_names)) if i not in self.artificial_vars] + [-1]
            self.tableau = self.tableau[:, cols_to_keep]
            self.var_names = [name for i, name in enumerate(self.var_names) if i not in self.artificial_vars]

            # Обновляем индексы базиса
            for i in range(len(self.basis)):
                if self.basis[i] in self.artificial_vars:
                    pass  # Теоретически сюда не дойдем, если решение есть
                else:
                    # Сдвигаем индекс из-за удаленных столбцов
                    self.basis[i] -= sum(1 for art in self.artificial_vars if art < self.basis[i])

        # Основная задача
        print("\nРешение основной задачи:")
        self.tableau[-1, :] = 0  # Очищаем строку Δ

        # Восстанавливаем целевую функцию
        for i in range(self.num_orig_vars):
            self.tableau[-1, i] = self.c[i]

        # Подстановка базисных переменных (выражение через свободные)
        for i in range(self.num_constraints):
            bas_col = self.basis[i]
            if self.tableau[-1, bas_col] != 0:
                self.tableau[-1, :] -= self.tableau[i, :] * self.tableau[-1, bas_col]

        self.solve_phase("Фаза 2")

        # Вывод ответа
        print("\nОтвет:")
        ans = [Fraction(0)] * self.num_orig_vars
        for i in range(self.num_constraints):
            if self.basis[i] < self.num_orig_vars:
                ans[self.basis[i]] = self.tableau[i, -1]

        ans_str = ", ".join(str(x) for x in ans)
        print(f"Оптимальная точка X* = ({ans_str})")

        z = self.tableau[-1, -1] if not self.maximize else -self.tableau[-1, -1]
        print(f"Значение целевой функции Z = {z}")



# Задача из Варианта №9
if __name__ == "__main__":
    # Целевая функция
    objective = [2, 1, 3, 2]  # Коэффициенты при x1, x2, x3, x4

    # Коэффициенты при переменных в левой части условий
    constraints = [
        [1, 2, 1, 0],  # x1 + 2x2 + x3 <= 11
        [1, 0, 1, 1],  # x1 + x3 + x4 = 8
        [0, 1, 0, 1]  # x2 + x4 >= 3
    ]

    # Свободные члены условий
    b = [11, 8, 3]

    # Знаки условий
    signs = ['<=', '=', '>=']

    # Создаем и запускаем решатель
    solver = SimplexSolver(objective, constraints, b, signs, maximize=True)
    solver.solve()