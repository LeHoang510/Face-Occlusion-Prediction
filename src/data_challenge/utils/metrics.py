import numpy as np


def weighted_mse(predictions: np.ndarray, ground_truths: np.ndarray) -> float:
    """Weighted MSE: w_i = 1/30 + GT_i."""
    weights = 1 / 30 + ground_truths
    return float(np.sum(weights * (predictions - ground_truths) ** 2) / np.sum(weights))


def compute_score(
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    genders: np.ndarray,
) -> tuple[float, float, float]:
    """Compute the challenge score.

    Score = (ErrF + ErrM) / 2 + |ErrF - ErrM|

    Returns:
        (score, err_female, err_male)
    """
    predictions = np.asarray(predictions, dtype=np.float32)
    ground_truths = np.asarray(ground_truths, dtype=np.float32)
    genders = np.asarray(genders, dtype=np.float32)

    female_mask = genders == 0.0
    male_mask = genders == 1.0

    err_f = weighted_mse(predictions[female_mask], ground_truths[female_mask])
    err_m = weighted_mse(predictions[male_mask], ground_truths[male_mask])
    score = (err_f + err_m) / 2 + abs(err_f - err_m)

    return score, err_f, err_m
