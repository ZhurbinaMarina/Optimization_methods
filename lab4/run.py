from __future__ import annotations

import argparse
import os
import sys

from engine import SCALE, Dynamics, DpSolver
from variant import ASSETS, CASE_TAG, PHASES, START

def de(x):
    return x / SCALE

def spell_move(move):
    chunks = []
    for a, q in zip(ASSETS, move):
        match q:
            case q if q > 0:
                chunks.append(f"{a.label}: покупка {q} пак. (+{q * a.lot:.0f} д.е.)")
            case q if q < 0:
                chunks.append(f"{a.label}: продажа {-q} пак. ({q * a.lot:.0f} д.е.)")
            case _:
                chunks.append(f"{a.label}: сделок нет")
    return "; ".join(chunks)

def path_rows(sol):
    head = (f"{'N':>3} {'сценарий':<9} {'P':>5} {'x1':>12} {'x2':>12} {'x3':>12} "
            f"{'w1':>8} {'w2':>8} {'w3':>8} {'D':>9} {'W4':>9}")
    lines = ["Траектории ситуаций (27 шт., по убыванию вероятности):", head,
             "-" * len(head)]
    n = 0
    for p in sorted(sol.paths, key=lambda r: -r.weight):
        n += 1
        tag = "-".join(CASE_TAG[c] for c in p.cases)
        moves = [f"({','.join(f'{q:+d}' for q in m)})" for m in p.moves]
        lines.append(f"{n:>3} {tag:<9} {p.weight:>5.3f} {moves[0]:>12} {moves[1]:>12} "
                     f"{moves[2]:>12} {p.gains[0]:>8.1f} {p.gains[1]:>8.1f} "
                     f"{p.gains[2]:>8.1f} {p.total:>9.1f} {p.final:>9.1f}")
    return lines

def render(sol, label):
    first = sol.paths[0].moves[0]
    layers = ", ".join(f"этап {t + 1} -> {sol.sizes[t]}"
                       for t in sorted(sol.sizes, reverse=True))
    lines = [
        "=" * 58,
        f"  КРИТЕРИЙ: {label}",
        "=" * 58,
        "Точность счёта: целые единицы 1/" + f"{SCALE} д.е. (без округлений)",
        f"Портфель на старте: ЦБ1 {START[0]:.0f}, ЦБ2 {START[1]:.0f}, "
        f"Деп {START[2]:.0f}, свободные деньги {START[3]:.0f} (W1 = {sum(START):.0f})",
        "",
        f"Оптимум в корне W1*(Xi_1) = {sol.root:.2f}",
        f"Средний доход E[D] = {sol.mean_d:.2f}",
        f"Худшая траектория D = {sol.worst_d:.2f}, лучшая D = {sol.best_d:.2f}",
        f"Слои состояний: {layers}",
        f"Единственность: в корне максимум на {sol.root_ties} управлениях; "
        f"на этапе 2 несколько оптимумов у {sol.ties_mid} состояний",
        "При равенстве берётся первое управление в порядке перебора",
        f"Время расчёта: {sol.seconds:.1f} с",
        "",
        f"Первая сделка x1* = ({','.join(f'{q:+d}' for q in first)})",
    ]
    for chunk in spell_move(first).split("; "):
        lines.append(f"  {chunk}")
    lines.append("")
    lines.extend(path_rows(sol))
    return "\n".join(lines)

def save(name, text, folder="output"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return path

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="ЛР4: управление портфелем за 3 этапа методом ДП, критерий Байеса",
        epilog="python run.py            # оптимум по ожиданию + база без сделок\n"
               "python run.py -w 4      # то же в 4 процесса")
    parser.add_argument("-w", "--workers", type=int, default=1,
                        help="число процессов для слоя этапа 2 (по умолчанию 1)")
    args = parser.parse_args(argv)
    try:
        solver = DpSolver(Dynamics(ASSETS, PHASES, START))
    except ValueError as err:
        print(f"Ошибка входных данных: {err}")
        sys.exit(1)
    print("Критерий Байеса: расчёт оптимальной стратегии", flush=True)
    sol = solver.run(workers=max(1, args.workers))
    suffix = "_multiprocess" if args.workers > 1 else ""
    text = render(sol, "Байес (математическое ожидание)")
    print(text)
    print(save(f"bayes{suffix}", text))
    print("Базовая стратегия (без сделок): расчёт", flush=True)
    base = solver.run(hold_only=True)
    text2 = render(base, "без сделок, оценка по Байесу")
    print(text2)
    print(save("hold", text2))

if __name__ == "__main__":
    main()
