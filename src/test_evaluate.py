from evaluate import entity_f05


def test_perfect_match():
    assert entity_f05({"A", "B"}, {"A", "B"}) == 1.0


def test_true_empty():
    assert entity_f05(set(), set()) == 1.0


def test_false_positive():
    assert entity_f05({"A"}, set()) == 0.0


def test_false_negative():
    assert entity_f05(set(), {"A"}) == 0.0


def test_partial_match():
    score = entity_f05({"A"}, {"A", "B"})
    assert 0 < score < 1


if __name__ == "__main__":
    test_perfect_match()
    test_true_empty()
    test_false_positive()
    test_false_negative()
    test_partial_match()

    print("ALL EVALUATION TESTS PASSED")

