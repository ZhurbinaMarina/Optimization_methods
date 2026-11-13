import numpy as np

class Rule:

    name = "Критерий"

    def grade(self, F, p):
        raise NotImplementedError

class Bayes(Rule):

    name = "Байес"

    def grade(self, F, p):
        return F @ np.asarray(p, dtype=float)

class Wald(Rule):

    name = "Вальд"

    def grade(self, F, p):
        return F.min(axis=1)

class Savage(Rule):

    name = "Сэвидж"

    def grade(self, F, p):
        regret = F.max(axis=0) - F
        return -regret.max(axis=1)

class Laplace(Rule):

    name = "Лаплас"

    def grade(self, F, p):
        return F.mean(axis=1)

class Hurwicz(Rule):

    def __init__(self, alpha=0.5):
        match alpha:
            case a if 0.0 <= a <= 1.0:
                self.alpha = float(a)
            case _:
                raise ValueError(f"alpha нужен в пределах от 0 до 1, а вышло {alpha}")
        self.name = f"Гурвиц (alpha={self.alpha:g})"

    def grade(self, F, p):
        return self.alpha * F.max(axis=1) + (1.0 - self.alpha) * F.min(axis=1)
