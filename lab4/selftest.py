from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import SCALE, Dynamics, DpSolver
from variant import Asset, Phase

FAILS = []

def check(num, title, inp, expected, actual, ok):
    mark = "СОВПАДАЕТ" if ok else "ОШИБКА"
    print(f"Тест {num}. {title}")
    print(f"  вход:       {inp}")
    print(f"  ожидаемо:   {expected}")
    print(f"  фактически: {actual}")
    if not ok:
        FAILS.append(num)
    print()

def brute(dyn, t, state):
    if t == len(dyn.phases):
        return 0.0
    U, F, Next = dyn.options(t, state)
    if U.shape[0] == 0:
        return float("-inf")
    best = float("-inf")
    k = 0
    while k < U.shape[0]:
        move = tuple(int(q) for q in U[k])
        fresh, purse = dyn.make_deals(state, move)
        acc = 0.0
        c = 0
        while c < len(dyn.p[t]):
            nxt = dyn.grow(fresh, purse, t, c)
            gain = (sum(nxt) - sum(state)) / SCALE
            acc += dyn.p[t][c] * (gain + brute(dyn, t + 1, nxt))
            c += 1
        if acc > best:
            best = acc
        k += 1
    return best

def solve(assets, phases, start):
    solver = DpSolver(Dynamics(assets, phases, start))
    return solver.run(), solver

def case_brute_small():
    assets = (Asset("A", lot=100.0, floor=100.0, fee=0.10),)
    phases = (Phase(p=(0.6, 0.4), k=((1.5,), (0.5,))),
              Phase(p=(0.5, 0.5), k=((1.2,), (0.9,))))
    start = (100.0, 200.0)
    sol, _ = solve(assets, phases, start)
    dyn = Dynamics(assets, phases, start)
    bf = brute(dyn, 0, dyn.start)
    ok = abs(sol.root - bf) < 1e-9 and abs(sol.root - 21.0) < 1e-9
    check(1, "Полный перебор всех политик на малом примере (2 этапа, 1 актив)",
          "старт (100; 200), лот 100, комиссия 10%, коэфф. 1.5/0.5 и 1.2/0.9",
          f"ДП = перебор = 21.00 д.е. (ручной расчёт)",
          f"ДП = {sol.root:.2f}, перебор = {bf:.2f}", ok)

def case_zero_resource():
    assets = (Asset("A", lot=25.0, floor=0.0, fee=0.10),)
    phases = (Phase(p=(1.0,), k=((1.5,),)),)
    sol, _ = solve(assets, phases, (0.0, 0.0))
    ok = abs(sol.root) < 1e-9 and sol.paths[0].moves[0] == (0,)
    check(2, "Нулевой объём ресурса",
          "старт (0; 0), лот 25, комиссия 10%",
          "доход 0, единственная политика - без сделок",
          f"корень {sol.root:.2f}, первое управление {sol.paths[0].moves[0]}", ok)

def case_one_asset_zero_returns():
    assets = (Asset("A", lot=25.0, floor=0.0, fee=0.10),)
    phases = (Phase(p=(0.5, 0.5), k=((1.0,), (1.0,))),
              Phase(p=(0.5, 0.5), k=((1.0,), (1.0,))))
    sol, _ = solve(assets, phases, (50.0, 50.0))
    hold_first = sol.paths[0].moves[0] == (0,)
    ok = abs(sol.root) < 1e-9 and hold_first
    check(3, "Нулевые доходности всех активов",
          "старт (50; 50), лот 25, комиссия 10%, все коэффициенты 1.00",
          "любая сделка несёт комиссию, оптимум - доход 0 без сделок",
          f"корень {sol.root:.2f}, первое управление {sol.paths[0].moves[0]}", ok)

def case_multiple_optima():
    assets = (Asset("A", lot=50.0, floor=0.0, fee=0.0),
              Asset("B", lot=50.0, floor=0.0, fee=0.0))
    phases = (Phase(p=(1.0,), k=((1.1, 1.1),)),)
    sol, _ = solve(assets, phases, (0.0, 0.0, 100.0))
    ok = abs(sol.root - 10.0) < 1e-9 and sol.root_ties >= 2
    check(4, "Несколько оптимальных планов",
          "два одинаковых актива без комиссий, старт (0; 0; 100), рост 1.10",
          "покупки (2;0), (1;1) и (0;2) равноценны: корень 10.00, альтернатив >= 2",
          f"корень {sol.root:.2f}, альтернатив в корне {sol.root_ties}", ok)

def case_all_in_one_asset():
    assets = (Asset("A", lot=20.0, floor=0.0, fee=0.0),)
    phases = (Phase(p=(1.0,), k=((2.0,),)),)
    sol, solver = solve(assets, phases, (0.0, 60.0))
    first = sol.paths[0].moves[0]
    bf = brute(Dynamics(assets, phases, (0.0, 60.0)), 0, Dynamics(assets, phases, (0.0, 60.0)).start)
    ok = abs(sol.root - 60.0) < 1e-9 and first == (3,) and abs(bf - sol.root) < 1e-9
    check(5, "Весь ресурс - в один актив",
          "один актив лот 20 без комиссий, старт (0; 60), рост 2.00",
          "оптимум: купить все 3 пакета (весь ресурс), доход 60.00; ДП = перебор",
          f"корень {sol.root:.2f}, первое управление {first}", ok)

def case_bad_inputs():
    assets = (Asset("A", lot=25.0, floor=0.0, fee=0.10),)
    ok1, msg1 = True, ""
    try:
        Dynamics(assets, (Phase(p=(1.0,), k=((1.5,),)),), (-100.0, 50.0))
        ok1 = False
    except ValueError as err:
        msg1 = str(err)
    ok2, msg2 = True, ""
    try:
        Dynamics(assets, (Phase(p=(0.6, 0.4), k=((1.2,),)),), (50.0, 50.0))
        ok2 = False
    except ValueError as err:
        msg2 = str(err)
    ok3, msg3 = True, ""
    try:
        Dynamics(assets, (Phase(p=(0.5, 0.2), k=((1.2,), (0.9,))),), (50.0, 50.0))
        ok3 = False
    except ValueError as err:
        msg3 = str(err)
    ok = ok1 and ok2 and ok3
    check(6, "Некорректные входные данные",
          "отрицательный ресурс; строка коэффициентов короче списка ситуаций; "
          "вероятности с суммой 0.7",
          "три отказа с пояснением, расчёт не запускается",
          f"отказ 1: {msg1}; отказ 2: {msg2}; отказ 3: {msg3}", ok)

def main():
    print("Самопроверка: полный перебор и тесты по п. 8.3 методички\n")
    case_brute_small()
    case_zero_resource()
    case_one_asset_zero_returns()
    case_multiple_optima()
    case_all_in_one_asset()
    case_bad_inputs()
    total = 6
    good = total - len(FAILS)
    print(f"Итог: пройдено {good}/{total}")
    if FAILS:
        print("Провалены тесты:", ", ".join(str(n) for n in FAILS))
        sys.exit(1)

if __name__ == "__main__":
    main()
