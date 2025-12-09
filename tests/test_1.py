# tests/test_sanity.py

def test_math_works():
    assert 1 + 1 == 2


def test_true_is_true():
    assert True


def test_string_contains():
    s = "edgerunner"
    assert "edge" in s
