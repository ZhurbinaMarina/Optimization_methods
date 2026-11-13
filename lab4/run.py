from __future__ import annotations

import argparse
import os

from engine import Dynamics, DpSolver
from rules import Bayes, Hurwicz, Laplace, Savage, Wald
from variant import ASSETS, CASE_TAG, START

W1 = sum(START)

RULES = {"bayes": Bayes, "laplace": Laplace, "wald": Wald,
         "hurwicz": Hurwicz, "savage": Savage}

STORE = {}

def make_rule(name, alpha=0.5):
    cls = RULES[name]
    return cls(alpha) if cls is Hurwicz else cls()

def fetch(name, alpha=0.5, hold_only=False):
    tag = (name, alpha, hold_only)
    if tag not in STORE:
        STORE[tag] = DpSolver(Dynamics()).run(make_rule(name, alpha), hold_only=hold_only)
    return STORE[tag]

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
    head = (f"{'N':>3} {'сценарий':<9} {'P':>5} {'x1':>11} {'x2':>11} {'x3':>11} "
            f"{'w1':>7} {'w2':>7} {'w3':>7} {'D':>8} {'W4':>8}")
    lines = ["Траектории ситуаций (27 шт., по убыванию вероятности):", head,
             "-" * len(head)]
    n = 0
    for p in sorted(sol.paths, key=lambda r: -r.weight):
        n += 1
        tag = "-".join(CASE_TAG[c] for c in p.cases)
        moves = [f"({','.join(f'{q:+d}' for q in m)})" for m in p.moves]
        lines.append(f"{n:>3} {tag:<9} {p.weight:>5.3f} {moves[0]:>11} {moves[1]:>11} "
                     f"{moves[2]:>11} {p.gains[0]:>7.1f} {p.gains[1]:>7.1f} "
                     f"{p.gains[2]:>7.1f} {p.total:>8.1f} {p.final:>8.1f}")
    return lines

def render(sol):
    first = sol.paths[0].moves[0]
    layers = ", ".join(f"этап {t + 1} -> {sol.sizes[t]}"
                       for t in sorted(sol.sizes, reverse=True))
    lines = [
        "=" * 54,
        f"  КРИТЕРИЙ: {sol.rule}",
        "=" * 54,
        f"Портфель на старте: ЦБ1 {START[0]:.0f}, ЦБ2 {START[1]:.0f}, "
        f"Деп {START[2]:.0f}, свободные деньги {START[3]:.0f} (W1 = {W1:.0f})",
        "",
        f"Оптимум в корне W1*(Xi_1) = {sol.root:.2f}",
        f"Средний доход E[D] = {sol.mean_d:.2f}",
        f"Худшая траектория D = {sol.worst_d:.2f}, лучшая D = {sol.best_d:.2f}",
        f"Слои состояний: {layers}",
        f"Время расчёта: {sol.seconds:.1f} с",
        "",
        f"Первая сделка x1* = ({','.join(f'{q:+d}' for q in first)})",
    ]
    for chunk in spell_move(first).split("; "):
        lines.append(f"  {chunk}")
    lines.append("")
    lines.extend(path_rows(sol))
    return "\n".join(lines)

def summary(alpha=0.5):
    rows = []
    for name in ("bayes", "laplace", "wald", "hurwicz", "savage"):
        a = alpha if name == "hurwicz" else 0.5
        sol = fetch(name, a)
        rows.append((sol.rule, sol.root, sol.mean_d, sol.worst_d, sol.best_d))
    base = fetch("bayes", hold_only=True)
    rows.append(("без сделок (по Байесу)", base.root, base.mean_d,
                 base.worst_d, base.best_d))
    head = f"{'критерий':<26}{'корень':>9}{'E[D]':>9}{'худший D':>10}{'лучший D':>10}"
    lines = ["Сводка по всем критериям (доходы в д.е.):", head, "-" * len(head)]
    for label, root, mean, worst, best in rows:
        lines.append(f"{label:<26}{root:>9.2f}{mean:>9.2f}{worst:>10.2f}{best:>10.2f}")
    return "\n".join(lines)

def save(name, text, folder="output"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return path

MENU = """\
Чем считать портфель?
  b - Байес: максимум среднего дохода
  l - Лаплас: все ситуации равновероятны
  w - Вальд: лучший из гарантированных исходов
  h - Гурвиц: смесь оптимизма и пессимизма (alpha)
  s - Сэвидж: наименьшее сожаление
  t - сводная таблица по всем критериям
Ответ (b/l/w/h/s/t): """

LETTERS = {"b": "bayes", "l": "laplace", "w": "wald", "h": "hurwicz", "s": "savage"}

def menu(alpha=0.5):
    choice = input(MENU).strip().lower()
    if choice == "t":
        text = summary(alpha)
        print(text)
        save("summary", text)
    elif choice in LETTERS:
        a = alpha
        if choice == "h":
            a = float(input("alpha от 0 до 1 (Enter = 0.5): ").strip() or "0.5")
        text = render(fetch(LETTERS[choice], a))
        print(text)
        save(LETTERS[choice], text)
    else:
        print("Такого пункта нет; подсказка: python run.py -h")

def cli(argv=None):
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="ЛР4: управление портфелем за 3 этапа методом динамического программирования",
        epilog="Примеры: python run.py bayes | python run.py hurwicz -a 0.3 | python run.py all")
    parser.add_argument("target", nargs="*",
                        help="критерии через пробел: bayes laplace wald hurwicz savage или all")
    parser.add_argument("-a", "--alpha", type=float, default=0.5,
                        help="коэффициент оптимизма для Гурвица (по умолчанию 0.5)")
    args = parser.parse_args(argv)
    if not args.target:
        menu(args.alpha)
    elif "all" in args.target:
        text = summary(args.alpha)
        print(text)
        save("summary", text)
    else:
        for name in args.target:
            text = render(fetch(name, args.alpha))
            print(text)
            save(name, text)

if __name__ == "__main__":
    cli()
