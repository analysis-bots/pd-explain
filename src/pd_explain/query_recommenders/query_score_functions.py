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
    if np.all(scores >= 0) and np.all(scores <= 1):
        return np.mean(scores)
    scores = np.array([(np.log10(1 + x) / (1 + np.log10(1 + x))) for x in scores])
    return np.mean(scores)
