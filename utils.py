"""
Utility functions including simple triangular/ trapezoid membership functions for fuzzy risk scoring.
"""


def triangular(x, a, b, c):
    """Triangular membership function with points a < b < c."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trapezoid(x, a, b, c, d):
    """Trapezoidal membership with a <= b <= c <= d."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def fuzzy_risk_score(glucose: float, bmi: float = None, age: float = None) -> float:
    """Compute a simple fuzzy risk score [0,1] combining glucose, bmi and age.
    - glucose: primary driver; membership functions chosen to reflect risk bands.
    - returns a fused degree where higher is higher risk.
    """
    # glucose memberships (mg/dL)
    normal = trapezoid(glucose, 0, 0, 90, 100)
    prediab = triangular(glucose, 90, 110, 126)
    diab = trapezoid(glucose, 110, 126, 300, 600)

    # bmi membership (if provided)
    if bmi is None:
        bmi_score = 0.5
    else:
        # simple mapping: normal <25, overweight 25-30, obese >30
        normal_b = trapezoid(bmi, 0, 0, 22, 25)
        over = triangular(bmi, 23, 27.5, 30)
        obese = trapezoid(bmi, 28, 30, 60, 100)
        bmi_score = max(over, obese*1.2, normal_b*0.5)

    # age factor (older age increases risk slightly)
    if age is None:
        age_factor = 1.0
    else:
        if age < 35:
            age_factor = 0.9
        elif age < 55:
            age_factor = 1.0
        else:
            age_factor = 1.1

    # fuse: weight glucose more heavily
    fused = (0.7 * diab + 0.2 * prediab + 0.1 * normal) * 0.8 + 0.2 * bmi_score
    fused = fused * age_factor
    # clip
    if fused < 0:
        fused = 0.0
    if fused > 1:
        fused = 1.0
    return fused


if __name__ == '__main__':
    # quick smoke test
    print(fuzzy_risk_score(85, bmi=23, age=30))
    print(fuzzy_risk_score(110, bmi=30, age=60))
    print(fuzzy_risk_score(140, bmi=35, age=50))
