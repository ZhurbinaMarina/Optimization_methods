from __future__ import annotations

import itertools
import math
import time
from collections import Counter

import numpy as np

from variant import ASSETS, CASH_STEP, PHASES, START

TOL = 1e-9
BAD = float("-inf")
BASE = 256

class Dynamics:

    def __init__(self, assets=ASSETS, phases=PHASES, cash_step=CASH_STEP):
        self.assets = tuple(assets)
        self.phases = tuple(phases)
        self.cash_step = float(cash_step)
        self.steps = tuple(a.step for a in self.assets) + (self.cash_step,)
        self.lots = tuple(a.lot for a in self.assets)
        self.floors = tuple(a.floor for a in self.assets)
        self.fees = tuple(a.fee for a in self.assets)
        self.mults = np.array([BASE ** i for i in range(len(self.steps))], dtype=np.int64)

    def to_nodes(self, values):
        return tuple(int(math.floor(v / g + 0.5)) for v, g in zip(values, self.steps))

    def to_money(self, nodes):
        return tuple(n * g for n, g in zip(nodes, self.steps))

    def pack(self, nodes):
        code, i = 0, len(nodes) - 1
        while i >= 0:
            code = code * BASE + nodes[i]
            i -= 1
        return code

    def unpack(self, code):
        digits, i = [], 0
        while i < len(self.steps):
            digits.append(code % BASE)
            code //= BASE
            i += 1
        return tuple(digits)

    def deal_cost(self, qty, asset):
        rate = (1.0 + asset.fee) if qty > 0 else (1.0 - asset.fee)
        return qty * asset.lot * rate

    def make_deals(self, values, move):
        held = values[:-1]
        purse = values[-1]
        fresh = tuple(s + q * a.lot for s, q, a in zip(held, move, self.assets))
        purse_left = purse - sum(self.deal_cost(q, a) for q, a in zip(move, self.assets))
        if purse_left < -TOL:
            return None
        j = 0
        while j < len(fresh):
            if fresh[j] < self.floors[j] - TOL:
                return None
            j += 1
        return fresh, purse_left

    def evolve(self, held, purse, t, case):
        k = self.phases[t].k[case]
        grown = [h * m for h, m in zip(held, k)]
        return self.to_nodes([*grown, purse])

    def lot_span(self, nodes):
        held = self.to_money(nodes[:-1])
        purse = nodes[-1] * self.cash_step
        spans = []
        for s, a in zip(held, self.assets):
            lo = math.ceil((a.floor - s) / a.lot - TOL)
            hi = math.floor(purse / (a.lot * (1.0 + a.fee)) + TOL)
            if lo > hi:
                return None
            spans.append((lo, hi))
        return tuple(spans)

    def all_moves(self, nodes):
        spans = self.lot_span(nodes)
        if spans is None:
            return np.zeros((0, len(self.assets)), dtype=np.int64)
        axes = [range(lo, hi + 1) for lo, hi in spans]
        return np.array(list(itertools.product(*axes)), dtype=np.int64)

    def options(self, t, nodes, with_next=True):
        n_case = len(self.phases[t].p)
        U = self.all_moves(nodes)
        if U.shape[0] == 0:
            return U, np.zeros((0, n_case)), np.zeros((0, n_case), dtype=np.int64)

        held = np.array(self.to_money(nodes[:-1]))
        purse = nodes[-1] * self.cash_step
        lot = np.array(self.lots)
        fee = np.array(self.fees)
        floor = np.array(self.floors)
        step = np.array(self.steps)

        M = U.astype(float)
        stream = M * lot * np.where(M > 0, 1.0 + fee, 1.0 - fee)
        purse_left = purse - stream.sum(axis=1)
        held_new = held[None, :] + M * lot[None, :]
        ok = (purse_left >= -TOL) & (held_new >= floor[None, :] - TOL).all(axis=1)
        U, purse_left, held_new = U[ok], purse_left[ok], held_new[ok]

        rich_before = held.sum() + purse
        F = np.empty((U.shape[0], n_case))
        codes = np.zeros((U.shape[0], n_case), dtype=np.int64)
        c = 0
        while c < n_case:
            k = np.array(self.phases[t].k[c])
            grown = np.concatenate([held_new * k[None, :], purse_left[:, None]], axis=1)
            cells = np.floor(grown / step[None, :] + 0.5).astype(np.int64)
            F[:, c] = (cells * step[None, :]).sum(axis=1) - rich_before
            if with_next:
                codes[:, c] = cells @ self.mults
            c += 1
        return U, F, codes

class Trajectory:

    def __init__(self, cases, weight, states, moves, gains, final):
        self.cases = cases
        self.weight = weight
        self.states = states
        self.moves = moves
        self.gains = gains
        self.total = float(sum(gains))
        self.final = final

class Outcome:

    def __init__(self, rule, root, start_nodes, paths, sizes, seconds):
        self.rule = rule
        self.root = root
        self.start_nodes = start_nodes
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

    def __init__(self, dyn, start=START):
        self.dyn = dyn
        self.start = dyn.to_nodes(start)
        self.w1 = sum(dyn.to_money(self.start))

    def run(self, rule, hold_only=False):
        stamp = time.perf_counter()
        worth, pick = self.tables(rule, hold_only)
        root = worth[(0, self.dyn.pack(self.start))]
        if root == BAD:
            raise RuntimeError("Из стартового портфеля не выходит ни одного допустимого управления")
        paths = self.walk(pick)
        sizes = dict(Counter(t for t, _ in worth))
        return Outcome(rule.name, root, self.start, paths, sizes,
                       time.perf_counter() - stamp)

    def tables(self, rule, hold_only):
        last = len(self.dyn.phases) - 1
        worth, pick = {}, {}
        shelf = {t: [] for t in range(last)}
        layer = [self.start]
        t = 0
        while t <= last:
            wave = set()
            for nodes in layer:
                key = (t, self.dyn.pack(nodes))
                U, F, codes = self.dyn.options(t, nodes, with_next=(t < last))
                if hold_only:
                    calm = np.where((U == 0).all(axis=1))[0]
                    if calm.size:
                        U, F, codes = U[calm], F[calm], codes[calm]
                if U.shape[0] == 0:
                    worth[key] = BAD
                    pick[key] = None
                    continue
                if t == last:
                    self.finish(worth, pick, key, rule, t, U, F, None)
                else:
                    shelf[t].append((key, U, F, codes))
                    c = 0
                    while c < codes.shape[1]:
                        wave.update(int(x) for x in np.unique(codes[:, c]))
                        c += 1
            if t == last:
                break
            layer = [self.dyn.unpack(x) for x in wave]
            t += 1

        t = last - 1
        while t >= 0:
            for key, U, F, codes in shelf[t]:
                cont = []
                c = 0
                while c < codes.shape[1]:
                    uniq, inv = np.unique(codes[:, c], return_inverse=True)
                    cont.append((uniq, inv))
                    c += 1
                self.finish(worth, pick, key, rule, t, U, F, cont)
            t -= 1
        return worth, pick

    def finish(self, worth, pick, key, rule, t, U, F, cont):
        if cont is None:
            total = F
        else:
            total = np.empty_like(F)
            c = 0
            while c < F.shape[1]:
                tail = np.array([worth[(t + 1, int(cd))] for cd in cont[c][0]])
                total[:, c] = F[:, c] + tail[cont[c][1]]
                c += 1
        marks = rule.grade(total, self.dyn.phases[t].p)
        best = int(np.argmax(marks))
        worth[key] = float(marks[best])
        pick[key] = tuple(int(q) for q in U[best])

    def walk(self, pick):
        paths = []
        end = len(self.dyn.phases)
        n_case = len(self.dyn.phases[0].p)
        todo = [(0, self.start, (), 1.0, (), (), ())]
        while todo:
            t, nodes, cases, weight, states, moves, gains = todo.pop()
            values = self.dyn.to_money(nodes)
            if t == end:
                paths.append(Trajectory(cases, weight, states, moves, gains,
                                        self.w1 + float(sum(gains))))
                continue
            move = pick[(t, self.dyn.pack(nodes))]
            held_after, purse_after = self.dyn.make_deals(values, move)
            s = 0
            while s < n_case:
                nxt = self.dyn.evolve(held_after, purse_after, t, s)
                gain = sum(self.dyn.to_money(nxt)) - sum(values)
                todo.append((t + 1, nxt, cases + (s,),
                             weight * self.dyn.phases[t].p[s],
                             states + (nodes,), moves + (move,), gains + (gain,)))
                s += 1
        return paths
