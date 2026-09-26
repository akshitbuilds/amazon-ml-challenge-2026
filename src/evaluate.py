from typing import Dict, Set


def parse_ids(value: str) -> Set[str]:
    if value is None:
        return set()

    value = str(value).strip()

    if not value:
        return set()

    return {
        x.strip()
        for x in value.split(",")
        if x.strip()
    }


def fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0 and recall == 0:
        return 0.0

    beta2 = beta * beta

    return (
        (1 + beta2) * precision * recall
        / (beta2 * precision + recall)
    )


def entity_f05(predicted: Set[str], actual: Set[str]) -> float:
    if not predicted and not actual:
        return 1.0

    if not predicted and actual:
        return 0.0

    if predicted and not actual:
        return 0.0

    tp = len(predicted & actual)

    precision = tp / len(predicted)
    recall = tp / len(actual)

    return fbeta(precision, recall, 0.5)


def macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]]
) -> float:

    scores = []

    for entity_id, actual in ground_truth.items():
        predicted = predictions.get(entity_id, set())
        scores.append(entity_f05(predicted, actual))

    if not scores:
        return 0.0

    return sum(scores) / len(scores)