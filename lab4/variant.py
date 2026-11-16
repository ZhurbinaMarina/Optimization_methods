class Asset:

    def __init__(self, label, lot, floor, fee):
        self.label = label
        self.lot = lot
        self.floor = floor
        self.fee = fee

class Phase:

    def __init__(self, p, k):
        self.p = p
        self.k = k

CASES = ("Благоприятная", "Нейтральная", "Негативная")
CASE_TAG = ("Б", "Н", "Нг")

BOND1 = Asset("ЦБ1", lot=25.0, floor=30.0, fee=0.04)
BOND2 = Asset("ЦБ2", lot=200.0, floor=150.0, fee=0.07)
DEPOSIT = Asset("Депозит", lot=100.0, floor=100.0, fee=0.05)

ASSETS = (BOND1, BOND2, DEPOSIT)

PHASES = (
    Phase(p=(0.60, 0.30, 0.10),
          k=((1.20, 1.10, 1.07),
             (1.05, 1.02, 1.03),
             (0.80, 0.95, 1.00))),
    Phase(p=(0.30, 0.20, 0.50),
          k=((1.40, 1.15, 1.01),
             (1.05, 1.00, 1.00),
             (0.60, 0.90, 1.00))),
    Phase(p=(0.40, 0.40, 0.20),
          k=((1.15, 1.12, 1.05),
             (1.05, 1.01, 1.01),
             (0.70, 0.94, 1.00))),
)

START = (100.0, 800.0, 400.0, 600.0)
