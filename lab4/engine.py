from __future__ import annotations

import itertools
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from variant import ASSETS, PHASES, START

SCALE = 10 ** 6
BAD = float("-inf")
PROGRESS_EVERY = 100
PARALLEL_MIN_STATES = 200

def _scaled(x, what):
    v = int(round(x * SCALE))
    if abs(x * SCALE - v) > 1e-6:
        raise ValueError(f"Значение {what} = {x} не выражается точно единицей 1/{SCALE} д.е.")
    return v

def _pct(x, what):
    v = int(round(x * 100))
    if abs(x * 100 - v) > 1e-6:
        raise ValueError(f"Значение {what} = {x} должно иметь не более двух знаков после запятой")
    return v

class Dynamics:

    def __init__(self, assets=ASSETS, phases=PHASES, start=START):
        self.assets = tuple(assets)
        self.phases = tuple(phases)
        self.raw_start = tuple(float(v) for v in start)
        n = len(self.assets)
        if len(start) != n + 1:
            raise ValueError(f"В начальном состоянии должно быть {n + 1} компоненты, получено {len(start)}")
        if any(v < 0 for v in start):
            raise ValueError(f"Отрицательный ресурс в начальном состоянии: {start}")
        for a in self.assets:
            if a.lot <= 0:
                raise ValueError(f"Пакет актива {a.label} должен быть положительным")
            if a.floor < 0:
                raise ValueError(f"Минимум актива {a.label} не может быть отрицательным")
            if not 0 <= a.fee < 1:
                raise ValueError(f"Комиссия актива {a.label} должна быть в пределах [0, 1)")
        for t, ph in enumerate(self.phases, 1):
            n_case = len(ph.p)
            if n_case == 0:
                raise ValueError(f"Этап {t}: список ситуаций пуст")
            if abs(sum(ph.p) - 1.0) > 1e-9:
                raise ValueError(f"Этап {t}: вероятности ситуаций должны давать в сумме 1")
            if len(ph.k) != n_case:
                raise ValueError(f"Этап {t}: строк коэффициентов {len(ph.k)} при {n_case} ситуациях (таблица неполна)")
            for s, row in enumerate(ph.k, 1):
                if len(row) != n:
                    raise ValueError(f"Этап {t}, ситуация {s}: коэффициентов {len(row)} при {n} активах (таблица неполна)")
        self.lots = tuple(_scaled(a.lot, f"пакет {a.label}") for a in self.assets)
        for a, lot in zip(self.assets, self.lots):
            if lot % 100:
                raise ValueError(f"Пакет актива {a.label} должен быть кратен 0,0001 д.е.")
        self.floors = tuple(_scaled(a.floor, f"минимум {a.label}") for a in self.assets)
        self.fees = tuple(_pct(a.fee, f"комиссия {a.label}") for a in self.assets)
        self.k = tuple(tuple(tuple(_pct(m, "коэффициент роста") for m in row)
                             for row in ph.k) for ph in self.phases)
        self.p = tuple(tuple(ph.p) for ph in self.phases)
        self.buy_unit = tuple(lot * (100 + fee) // 100
                              for lot, fee in zip(self.lots, self.fees))
        self.sell_unit = tuple(lot * (100 - fee) // 100
                               for lot, fee in zip(self.lots, self.fees))
        self.start = tuple(_scaled(v, "начальное значение") for v in start)
        self.lot_arr = np.array(self.lots, dtype=np.int64)
        self.buy_arr = np.array(self.buy_unit, dtype=np.int64)
        self.sell_arr = np.array(self.sell_unit, dtype=np.int64)
        self.floor_arr = np.array(self.floors, dtype=np.int64)
        self.k_arr = tuple(tuple(np.array(row, dtype=np.int64) for row in ph)
                           for ph in self.k)
        self.p_arr = tuple(np.array(pp, dtype=float) for pp in self.p)

    def make_deals(self, state, move):
        fresh = tuple(s + q * lot for s, q, lot in zip(state, move, self.lots))
        cost = 0
        for q, up, un in zip(move, self.buy_unit, self.sell_unit):
            cost += q * up if q > 0 else q * un
        purse = state[-1] - cost
        if purse < 0:
            return None
        for f, floor in zip(fresh, self.floors):
            if f < floor:
                return None
        return fresh, purse

    def grow(self, fresh, purse, t, case):
        row = self.k[t][case]
        for s in fresh:
            if s % 100:
                raise ValueError("Внутренняя ошибка: значение не делится на 100 при росте")
        return tuple(s * m // 100 for s, m in zip(fresh, row)) + (purse,)

    def budget(self, state):
        total = state[-1]
        for s, floor, fee in zip(state, self.floors, self.fees):
            total += (s - floor) * (100 - fee) // 100
        return total

    def bounds(self, state):
        budget = self.budget(state)
        spans = []
        for s, floor, lot, unit in zip(state, self.floors, self.lots, self.buy_unit):
            lo = (floor - s + lot - 1) // lot
            hi = budget // unit
            if lo > hi:
                return None
            spans.append((lo, hi))
        return tuple(spans)

    def enumerate_moves(self, state):
        n_assets = len(self.lots)
        spans = self.bounds(state)
        if spans is None:
            return np.zeros((0, n_assets), dtype=np.int64)
        if n_assets == 3:
            return self._slices_three(state, spans)
        axes = [range(lo, hi + 1) for lo, hi in spans]
        if any(len(rng) == 0 for rng in axes):
            return np.zeros((0, n_assets), dtype=np.int64)
        U = np.array(list(itertools.product(*axes)), dtype=np.int64)
        spend = U * np.where(U > 0, self.buy_arr, self.sell_arr)
        purse_left = state[-1] - spend.sum(axis=1)
        fresh = np.array(state[:-1], dtype=np.int64)[None, :] + U * self.lot_arr[None, :]
        ok = (purse_left >= 0) & (fresh >= self.floor_arr[None, :]).all(axis=1)
        return U[ok]

    def _slices_three(self, state, spans):
        (lo1, _), (lo2, hi2), (lo3, hi3) = spans
        u2v = np.arange(lo2, hi2 + 1, dtype=np.int64)
        u3v = np.arange(lo3, hi3 + 1, dtype=np.int64)
        if u2v.size == 0 or u3v.size == 0:
            return np.zeros((0, 3), dtype=np.int64)
        f2 = np.where(u2v > 0, u2v * self.buy_arr[1], u2v * self.sell_arr[1])
        f3 = np.where(u3v > 0, u3v * self.buy_arr[2], u3v * self.sell_arr[2])
        residual = int(state[-1]) - f2[:, None] - f3[None, :]
        unit1 = np.where(residual >= 0, self.buy_arr[0], self.sell_arr[0])
        u1hi = residual // unit1
        lens = np.maximum(u1hi - lo1 + 1, 0)
        flat = lens.ravel()
        total = int(flat.sum())
        if total == 0:
            return np.zeros((0, 3), dtype=np.int64)
        ends = np.cumsum(flat)
        starts = ends - flat
        idx = np.arange(total, dtype=np.int64) - np.repeat(starts, flat)
        u1 = lo1 + idx
        u2 = np.repeat(np.repeat(u2v, u3v.size), flat)
        u3 = np.repeat(np.tile(u3v, u2v.size), flat)
        return np.stack([u1, u2, u3], axis=1)

    def options(self, t, state, with_next=True):
        n_case = len(self.p[t])
        M = self.enumerate_moves(state)
        if M.shape[0] == 0:
            return M, np.zeros((0, n_case), dtype=np.int64), None
        held = np.array(state[:-1], dtype=np.int64)
        purse = int(state[-1])

        spend = M * np.where(M > 0, self.buy_arr, self.sell_arr)
        purse_left = purse - spend.sum(axis=1)
        fresh = held[None, :] + M * self.lot_arr[None, :]
        if (fresh % 100).any():
            raise ValueError("Внутренняя ошибка: потеря точности в переходе")

        rich = int(held.sum() + purse)
        F = np.empty((M.shape[0], n_case), dtype=np.int64)
        Next = np.empty((M.shape[0], n_case, len(self.lots) + 1), dtype=np.int64) \
            if with_next else None
        c = 0
        while c < n_case:
            row = self.k_arr[t][c]
            grown = fresh * row[None, :] // 100
            F[:, c] = grown.sum(axis=1) + purse_left - rich
            if with_next:
                Next[:, c, :] = np.concatenate([grown, purse_left[:, None]], axis=1)
            c += 1
        return M, F, Next

    def hold_row(self, t, state):
        zeros = (0,) * len(self.lots)
        got = self.make_deals(state, zeros)
        if got is None:
            return None
        fresh, purse = got
        rich = sum(state)
        row, nxts = [], []
        c = 0
        while c < len(self.p[t]):
            nxt = self.grow(fresh, purse, t, c)
            row.append(sum(nxt) - rich)
            nxts.append(nxt)
            c += 1
        return np.array([row], dtype=np.int64), tuple(nxts)

def _leaf_value(dyn, state, hold_only):
    last = len(dyn.phases) - 1
    if hold_only:
        got = dyn.hold_row(last, state)
        if got is None:
            return BAD
        F, _ = got
    else:
        _, F, _ = dyn.options(last, state, with_next=False)
        if F.shape[0] == 0:
            return BAD
    scores = F.astype(float) @ dyn.p_arr[last]
    return float(scores[int(np.argmax(scores))])

def _eval_mid(dyn, state, hold_only):
    t = len(dyn.phases) - 2
    n_assets = len(dyn.lots)
    if hold_only:
        got = dyn.hold_row(t, state)
        if got is None:
            return BAD, None, 0, 0
        F, nxts = got
        U = np.zeros((1, n_assets), dtype=np.int64)
        Next = np.array([nxts], dtype=np.int64)
    else:
        U, F, Next = dyn.options(t, state)
        if U.shape[0] == 0:
            return BAD, None, 0, 0
    Ff = F.astype(float)
    calls = 0
    c = 0
    while c < F.shape[1]:
        uniq, inv = np.unique(Next[:, c, :], axis=0, return_inverse=True)
        calls += len(uniq)
        tails = np.array([_leaf_value(dyn, tuple(int(x) for x in row), hold_only)
                          for row in uniq])
        Ff[:, c] = Ff[:, c] + tails[inv]
        c += 1
    scores = Ff @ dyn.p_arr[t]
    best = int(np.argmax(scores))
    ties = int((scores == scores[best]).sum())
    return float(scores[best]), tuple(int(q) for q in U[best]), ties, calls

_POOL = {"dyn": None, "hold": False}

def _pool_init(assets, phases, start, hold_only):
    _POOL["dyn"] = Dynamics(assets, phases, start)
    _POOL["hold"] = hold_only

def _pool_eval(state):
    return _eval_mid(_POOL["dyn"], state, _POOL["hold"])

class Trajectory:

    def __init__(self, cases, weight, states, moves, gains, w1):
        self.cases = cases
        self.weight = weight
        self.states = states
        self.moves = moves
        self.gains = tuple(g / SCALE for g in gains)
        self.total = float(sum(self.gains))
        self.final = w1 + self.total

class Outcome:

    def __init__(self, root_scaled, root_ties, ties_mid, paths, sizes, seconds):
        self.root = root_scaled / SCALE
        self.root_ties = root_ties
        self.ties_mid = ties_mid
        self.paths = paths
        self.sizes = sizes
        self.seconds = seconds

    @property
    def mean_d(self):
        return sum(t.weight * t.total for t in self.paths)

    @property
    def worst_d(self):
        return min(t.total for t in self.paths)

    @property
    def best_d(self):
        return max(t.total for t in self.paths)

class DpSolver:

    def __init__(self, dyn):
        self.dyn = dyn
        self.start = dyn.start
        self.w1 = sum(dyn.start) / SCALE

    def run(self, hold_only=False, workers=1):
        stamp = time.perf_counter()
        worth, pick, root_ties, ties_mid, leaf_calls = \
            self.tables(hold_only, stamp, workers)
        root = worth.get((0, self.start), BAD)
        if root == BAD:
            raise RuntimeError("Из стартового состояния нет допустимых управлений")
        paths = self.walk(pick, hold_only)
        sizes = dict(Counter(t for t, _ in worth))
        sizes[len(self.dyn.phases) - 1] = leaf_calls
        return Outcome(root, root_ties, ties_mid, paths, sizes,
                       time.perf_counter() - stamp)

    def leaf_move(self, state):
        dyn = self.dyn
        t = len(dyn.phases) - 1
        U, F, _ = dyn.options(t, state, with_next=False)
        if U.shape[0] == 0:
            raise RuntimeError("Траектория оптимальной политики попала в состояние без допустимых управлений")
        scores = F.astype(float) @ dyn.p_arr[t]
        return tuple(int(q) for q in U[int(np.argmax(scores))])

    def tables(self, hold_only, stamp, workers=1):
        dyn = self.dyn
        last = len(dyn.phases) - 1
        n_assets = len(dyn.lots)
        worth, pick = {}, {}
        ties_mid = 0
        leaf_calls = 0
        root_ties = 0

        if last == 0:
            value, move, root_ties, calls = self._root_only(hold_only)
            worth[(0, self.start)] = value
            pick[(0, self.start)] = move
            return worth, pick, root_ties, 0, calls

        layers = {0: [self.start]}
        wave = set()
        if hold_only:
            got = dyn.hold_row(0, self.start)
            if got is not None:
                wave.update(got[1])
        else:
            U0, F0, Next0 = dyn.options(0, self.start)
            if U0.shape[0]:
                c = 0
                while c < Next0.shape[1]:
                    for row in np.unique(Next0[:, c, :], axis=0):
                        wave.add(tuple(int(x) for x in row))
                    c += 1
        layers[1] = sorted(wave)
        print(f"  слой этапа 2: состояний {len(layers[1])}", flush=True)

        def track_ties(t, ties):
            nonlocal root_ties, ties_mid
            if t == 0:
                root_ties = ties
            elif ties > 1:
                ties_mid += 1

        def progress(t, done, total, started):
            if total > PROGRESS_EVERY and (done % PROGRESS_EVERY == 0 or done == total):
                spent = time.perf_counter() - stamp
                rate = (time.perf_counter() - started) / done
                print(f"  этап {t + 1}: выполнено {done}/{total}, "
                      f"осталось ~{rate * (total - done) / 60:.0f} мин",
                      flush=True)

        t = last - 1
        states = layers[last - 1]
        total = len(states)
        print(f"  этап {t + 1}: состояний {total}", flush=True)
        started = time.perf_counter()
        if workers > 1 and total >= PARALLEL_MIN_STATES:
            done = 0
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_pool_init,
                                     initargs=(dyn.assets, dyn.phases,
                                               dyn.raw_start, hold_only)) as pool:
                for state, (value, move, ties, calls) in \
                        zip(states, pool.map(_pool_eval, states, chunksize=16)):
                    done += 1
                    progress(t, done, total, started)
                    worth[(t, state)], pick[(t, state)] = value, move
                    track_ties(t, ties)
                    leaf_calls += calls
        else:
            done = 0
            for state in states:
                done += 1
                progress(t, done, total, started)
                value, move, ties, calls = _eval_mid(dyn, state, hold_only)
                worth[(t, state)], pick[(t, state)] = value, move
                track_ties(t, ties)
                leaf_calls += calls

        t = last - 2
        while t >= 0:
            states = layers.get(t, [])
            total = len(states)
            done = 0
            started = time.perf_counter()
            print(f"  этап {t + 1}: состояний {total}", flush=True)
            for state in states:
                done += 1
                progress(t, done, total, started)
                key = (t, state)
                if hold_only:
                    got = dyn.hold_row(t, state)
                    if got is None:
                        worth[key], pick[key] = BAD, None
                        continue
                    F, nxts = got
                    U = np.zeros((1, n_assets), dtype=np.int64)
                    Next = np.array([nxts], dtype=np.int64)
                else:
                    U, F, Next = dyn.options(t, state)
                    if U.shape[0] == 0:
                        worth[key], pick[key] = BAD, None
                        continue
                Ff = F.astype(float)
                c = 0
                while c < F.shape[1]:
                    uniq, inv = np.unique(Next[:, c, :], axis=0, return_inverse=True)
                    tails = np.array([worth[(t + 1, tuple(int(x) for x in row))]
                                      for row in uniq])
                    Ff[:, c] = Ff[:, c] + tails[inv]
                    c += 1
                scores = Ff @ dyn.p_arr[t]
                best = int(np.argmax(scores))
                worth[key] = float(scores[best])
                pick[key] = tuple(int(q) for q in U[best])
                track_ties(t, int((scores == scores[best]).sum()))
            t -= 1
        return worth, pick, root_ties, ties_mid, leaf_calls

    def _root_only(self, hold_only):
        dyn = self.dyn
        if hold_only:
            got = dyn.hold_row(0, self.start)
            if got is None:
                return BAD, None, 0, 0
            F, _ = got
            scores = F.astype(float) @ dyn.p_arr[0]
            best = int(np.argmax(scores))
            return float(scores[best]), (0,) * len(dyn.lots), \
                int((scores == scores[best]).sum()), 1
        U, F, _ = dyn.options(0, self.start, with_next=False)
        if U.shape[0] == 0:
            return BAD, None, 0, 0
        scores = F.astype(float) @ dyn.p_arr[0]
        best = int(np.argmax(scores))
        return float(scores[best]), tuple(int(q) for q in U[best]), \
            int((scores == scores[best]).sum()), 1

    def walk(self, pick, hold_only=False):
        dyn = self.dyn
        paths = []
        end = len(dyn.phases)
        last = end - 1
        n_case = len(dyn.p[0])
        todo = [(0, self.start, (), 1.0, (), (), ())]
        while todo:
            t, state, cases, weight, states, moves, gains = todo.pop()
            if t == end:
                paths.append(Trajectory(cases, weight, states, moves, gains, self.w1))
                continue
            if t == last and hold_only:
                move = (0,) * len(dyn.lots)
            elif t == last:
                move = self.leaf_move(state)
            else:
                move = pick[(t, state)]
            fresh, purse = dyn.make_deals(state, move)
            s = 0
            while s < n_case:
                nxt = dyn.grow(fresh, purse, t, s)
                gain = sum(nxt) - sum(state)
                todo.append((t + 1, nxt, cases + (s,), weight * dyn.p[t][s],
                             states + (state,), moves + (move,), gains + (gain,)))
                s += 1
        return paths
