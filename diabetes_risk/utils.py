"""
Utility functions including simple triangular/ trapezoid membership functions for fuzzy risk scoring.
"""

import numpy as np


def triangular(x, a, b, c):
    """Triangular membership function with points a < b < c."""
    if not np.isfinite(x):
        return 0.0
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trapezoid(x, a, b, c, d):
    """Trapezoidal membership with a <= b <= c <= d."""
    if not np.isfinite(x):
        return 0.0
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a == b and x <= b:
        return 1.0
    if c == d and x >= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def fuzzy_risk_assessment(glucose: float = None, bmi: float = None, age: float = None,
                          family_history: bool = None) -> dict:
    """Mamdani-style rule inference with centroid defuzzification.

    Membership breakpoints are prototype settings, not clinically validated cutoffs.
    """
    g = ({
        'normal': trapezoid(glucose, 0, 0, 90, 100),
        'elevated': trapezoid(glucose, 90, 100, 125, 140),
        'high': trapezoid(glucose, 120, 126, 600, 600),
    } if glucose is not None else None)
    if bmi is None:
        b = {'normal': 0.5, 'overweight': 0.5, 'high': 0.5}
    else:
        b = {
            'normal': trapezoid(bmi, 0, 0, 22, 25),
            'overweight': trapezoid(bmi, 23, 25, 29, 31),
            'high': trapezoid(bmi, 29, 32, 100, 100),
        }
    if age is None:
        a = {'younger': 0.5, 'older': 0.5}
    else:
        a = {
            'younger': trapezoid(age, 0, 0, 30, 45),
            'older': trapezoid(age, 40, 60, 120, 120),
        }
    if family_history is None or (isinstance(family_history, (float, np.floating)) and np.isnan(family_history)):
        family = 0.5
    elif isinstance(family_history, str):
        normalized_family = family_history.strip().lower()
        family = (1.0 if normalized_family in {'yes', 'true', 'present', '1'}
                  else 0.0 if normalized_family in {'no', 'false', 'absent', '0'} else 0.5)
    else:
        family = float(bool(family_history))

    # Antecedents use minimum for AND and maximum for OR; consequents are fuzzy sets.
    if g is None:
        # Glucose is omitted for the current glucose-derived target to prevent
        # leakage through the fuzzy branch as well as through the ML features.
        rule_strengths = {
            'low': min(b['normal'], a['younger'], 1.0 - family),
            'moderate': max(b['overweight'], a['older']),
            'high': max(b['high'], min(b['overweight'], family),
                        min(b['high'], a['older'])),
        }
    else:
        rule_strengths = {
            'low': min(g['normal'], b['normal'], a['younger']),
            'moderate': max(g['elevated'], b['overweight'], min(g['normal'], b['high'], a['older'])),
            'high': max(g['high'], b['high'], min(g['elevated'], family),
                        min(g['elevated'], b['overweight'], a['older'])),
        }
    universe = np.linspace(0.0, 1.0, 1001)
    output_sets = {
        'low': np.array([trapezoid(x, 0.0, 0.0, 0.2, 0.45) for x in universe]),
        'moderate': np.array([triangular(x, 0.25, 0.5, 0.75) for x in universe]),
        'high': np.array([trapezoid(x, 0.55, 0.8, 1.0, 1.0) for x in universe]),
    }
    aggregated = np.maximum.reduce([
        np.minimum(rule_strengths[name], output_sets[name]) for name in rule_strengths
    ])
    score = float(np.trapz(universe * aggregated, universe) /
                  np.trapz(aggregated, universe)) if aggregated.any() else 0.5
    dimensions = [b, a] + ([g] if g is not None else [])
    uncertainty = float(np.mean([1.0 - max(values.values()) for values in dimensions]))
    return {
        'score': min(1.0, max(0.0, score)),
        'rule_strengths': rule_strengths,
        'memberships': {'glucose': g, 'bmi': b, 'age': a, 'family_history': family},
        'uncertainty': uncertainty,
    }


def fuzzy_risk_score(glucose: float, bmi: float = None, age: float = None,
                     family_history: bool = None) -> float:
    """Return the [0, 1] centroid output of the fuzzy rule system."""
    return fuzzy_risk_assessment(glucose, bmi, age, family_history)['score']


if __name__ == '__main__':
    # quick smoke test
    print(fuzzy_risk_score(85, bmi=23, age=30))
    print(fuzzy_risk_score(110, bmi=30, age=60))
    print(fuzzy_risk_score(140, bmi=35, age=50))
