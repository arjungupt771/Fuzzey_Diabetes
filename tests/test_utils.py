from diabetes_risk.utils import triangular, trapezoid, fuzzy_risk_score


def test_triangular_center():
    assert triangular(5, 0, 5, 10) == 1.0


def test_triangular_edges():
    assert triangular(0, 0, 5, 10) == 0.0
    assert triangular(10, 0, 5, 10) == 0.0


def test_trapezoid_middle():
    assert trapezoid(5, 0, 3, 7, 10) == 1.0


def test_fuzzy_scores_range():
    s1 = fuzzy_risk_score(85, bmi=22, age=30)
    s2 = fuzzy_risk_score(110, bmi=28, age=60)
    s3 = fuzzy_risk_score(140, bmi=35, age=50)
    for s in (s1, s2, s3):
        assert s >= 0.0 and s <= 1.0
