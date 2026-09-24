import numpy as np
from fractions import Fraction


class SimplexSolver:
    def __init__(self, objective, constraints, b, signs, maximize=True, free_vars=None):
        self.orig_obj = [Fraction(x) for x in objective]
        self.maximize = maximize
        self.constraints = [[Fraction(x) for x in row] for row in constraints]
        self.b = [Fraction(x) for x in b]
        self.signs = signs

        # Отслеживаем исходное количество переменных и их знаки
        self.actual_orig_vars_count = len(objective)
        self.free_vars_flags = free_vars if free_vars else [False] * self.actual_orig_vars_count

        self.num_orig_vars = 0  # Будет пересчитано с учетом свободных переменных
        self.num_constraints = len(b)

        self.tableau = None
        self.basis = []
        self.artificial_vars = []
        self.var_names = []
        self.var_mapping = {}  # Словарь для сборки свободных переменных обратно в ответе

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
        # Если переменная не ограничена в знаке, разбиваем её на две
        expanded_obj = []
        expanded_constraints = [[] for _ in range(self.num_constraints)]

        idx = 0
        for i in range(self.actual_orig_vars_count):
            expanded_obj.append(self.orig_obj[i])
            for r in range(self.num_constraints):
                expanded_constraints[r].append(self.constraints[r][i])
            self.var_names.append(f"x{i + 1}")

            if self.free_vars_flags[i]:
                # Добавляем отрицательную часть переменной
                expanded_obj.append(-self.orig_obj[i])
                for r in range(self.num_constraints):
                    expanded_constraints[r].append(-self.constraints[r][i])
                self.var_names.append(f"x{i + 1}'")
                self.var_mapping[i] = (idx, idx + 1)
                idx += 2
            else:
                self.var_mapping[i] = (idx, None)
                idx += 1

        self.num_orig_vars = len(expanded_obj)
        self.constraints = expanded_constraints

        # Приведение к каноническому виду
        if self.maximize:
            self.c = [-x for x in expanded_obj]
        else:
            self.c = list(expanded_obj)

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
        # Пересчитываем таблицу
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
            print(
                f"-> Входит в базис: переменная {self.var_names[pivot_col]}, Выходит: переменная {self.var_names[self.basis[pivot_row]]}")
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

            if self.tableau[-1, -1] < 0:
                raise Exception("Система ограничений несовместна (пустое множество решений)")

            # Если искусственная переменная осталась в базисе, выгоняем её или удаляем строку
            rows_to_delete = []
            for i in range(self.num_constraints):
                if self.basis[i] in self.artificial_vars:
                    swapped = False
                    # Ищем любую неискусственную переменную, чтобы сделать pivot
                    for j in range(len(self.var_names)):
                        if j not in self.artificial_vars and self.tableau[i, j] != 0:
                            print(
                                f"Вырожденный случай: принудительно выводим {self.var_names[self.basis[i]]}, вводим {self.var_names[j]}")
                            self.pivot(i, j)
                            swapped = True
                            break
                    if not swapped:
                        # Если все неискусственные коэффициенты равны 0, значит уравнение было избыточным
                        print(f"Вырожденный случай: удаляем избыточную (линейно зависимую) строку {i}")
                        rows_to_delete.append(i)

            # Удаляем избыточные строки с конца (чтобы не сбить индексы)
            for i in reversed(rows_to_delete):
                self.tableau = np.delete(self.tableau, i, axis=0)
                self.basis.pop(i)
                self.num_constraints -= 1

            # Удаляем столбцы искусственных переменных
            cols_to_keep = [i for i in range(len(self.var_names)) if i not in self.artificial_vars] + [-1]
            self.tableau = self.tableau[:, cols_to_keep]
            self.var_names = [name for i, name in enumerate(self.var_names) if i not in self.artificial_vars]

            # Сдвигаем индексы базиса из-за удаленных колонок
            for i in range(len(self.basis)):
                shift = sum(1 for art in self.artificial_vars if art < self.basis[i])
                self.basis[i] -= shift

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
        expanded_ans = [Fraction(0)] * self.num_orig_vars
        for i in range(self.num_constraints):
            if self.basis[i] < self.num_orig_vars:
                expanded_ans[self.basis[i]] = self.tableau[i, -1]

        # Собираем свободные переменные обратно
        final_ans = []
        for i in range(self.actual_orig_vars_count):
            pos_idx, neg_idx = self.var_mapping[i]
            val = expanded_ans[pos_idx]
            if neg_idx is not None:
                val -= expanded_ans[neg_idx]
            final_ans.append(val)

        ans_str = ", ".join(str(x) for x in final_ans)
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

    # Флаги свободных переменных (по умолчанию все False). Оставлено для универсальности
    free_vars = [False, False, False, False]

    # Создаем и запускаем решатель
    solver = SimplexSolver(objective, constraints, b, signs, maximize=True, free_vars=free_vars)
    solver.solve()