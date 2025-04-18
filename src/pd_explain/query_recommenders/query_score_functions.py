import numpy as np


def score_queries(scores: dict) -> float:
    """
    Gives unified score for a list of scores.
    The scores should be those produced when computing a measure for a query.
    If the scores are between 0 and 1, the score is the mean of the scores.
    Otherwise, the scores have a logarithmic transformation applied to it before taking the mean.

    :param scores: the score dict returned by the measure.
    :return: A float representing the score of the query.
    """
    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: -item[1])}
    scores = np.array([v for k, v in scores.items()][:4])
    if any(x > 1 for x in scores):
        scores = np.array([(np.log10(1 + x) / (1 + np.log10(1 + np.max(scores)))) for x in scores])
    # Return the geometric mean of the scores
    return np.pow(np.prod(scores), 1 / len(scores))
