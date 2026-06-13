"""Train/validation split strategies.

Why this exists
---------------
The challenge metric is gender-balanced by construction:

    score = (ErrF + ErrM) / 2 + |ErrF - ErrM|

and the *test* set is ~50/50 female/male, while the *train* pool is ~32/68.
Females also carry ~2x the occlusion of males, so they are both harder and
more heavily weighted (w = 1/30 + GT). A plain random 10% validation split
therefore estimates ErrF on a small, lucky ~32% female subsample, which makes
the validation score an optimistic and noisy predictor of the leaderboard.

`stratified_gender` builds a validation set whose gender balance mirrors the
test set (default 50/50) and whose per-gender occlusion distribution matches the
training pool (occlusion-stratified, deterministic). The resulting val score is
a faithful, low-variance stand-in for the leaderboard, so model selection and
hyper-parameter tuning optimise the real objective instead of a lucky proxy.

The default `random` strategy is handled by the callers (unchanged
`torch.random_split`); this module only implements the stratified path so that
existing configs keep byte-identical behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _occlusion_stratified_pick(
    group_idx: np.ndarray, target_vals: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Pick exactly ``k`` indices from ``group_idx``, spread across occlusion.

    The group is sorted by occlusion and cut into ``k`` contiguous strata; one
    random member is drawn from each stratum. This yields a subsample whose
    occlusion distribution matches the group (low variance on the per-group
    weighted MSE) while staying deterministic given ``rng``.
    """
    m = len(group_idx)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k >= m:
        return group_idx.astype(np.int64, copy=True)

    order = np.argsort(target_vals, kind="stable")
    ordered = group_idx[order]
    edges = np.linspace(0, m, num=k + 1).astype(int)
    out = np.empty(k, dtype=np.int64)
    for i in range(k):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            hi = lo + 1
        out[i] = ordered[rng.integers(lo, min(hi, m))]
    return out


def stratified_gender_split(
    df: pd.DataFrame,
    val_split: float,
    seed: int,
    val_female_ratio: float | None = 0.5,
    gender_col: str = "gender",
    target_col: str = "FaceOcclusion",
) -> tuple[list[int], list[int]]:
    """Return ``(train_indices, val_indices)`` for a gender-stratified split.

    Args:
        df: training dataframe (needs ``gender`` and ``FaceOcclusion`` columns).
        val_split: fraction of rows that go to validation (e.g. 0.1).
        seed: RNG seed — the split is fully deterministic given this seed.
        val_female_ratio: target female fraction *inside the val set*.
            0.5 mirrors the gender-balanced test set (recommended).
            ``None`` preserves the training female ratio (variance reduction
            only, no rebalancing).

    The two gender groups are each sampled occlusion-stratified so the val set
    is representative across the occlusion range; everything else is train.
    """
    n = len(df)
    val_size = int(n * val_split)
    rng = np.random.default_rng(seed)

    gender = df[gender_col].to_numpy()
    target = df[target_col].to_numpy()
    fem = np.where(gender == 0.0)[0]
    male = np.where(gender == 1.0)[0]
    if fem.size == 0 or male.size == 0:
        raise ValueError("stratified_gender split requires both genders to be present")

    if val_female_ratio is None:
        val_female_ratio = fem.size / n
    n_val_f = int(round(val_size * float(val_female_ratio)))
    n_val_m = val_size - n_val_f
    n_val_f = min(n_val_f, fem.size)
    n_val_m = min(n_val_m, male.size)

    val_f = _occlusion_stratified_pick(fem, target[fem], n_val_f, rng)
    val_m = _occlusion_stratified_pick(male, target[male], n_val_m, rng)
    val_idx = np.concatenate([val_f, val_m])

    val_mask = np.zeros(n, dtype=bool)
    val_mask[val_idx] = True
    train_idx = np.where(~val_mask)[0]

    rng.shuffle(val_idx)
    return train_idx.tolist(), val_idx.tolist()


def make_val_indices(
    df: pd.DataFrame, data_cfg: dict, seed: int
) -> tuple[list[int], list[int]]:
    """Dispatch to the configured non-random split strategy.

    Reads ``data.val_split_strategy`` (and ``data.val_female_ratio`` for the
    stratified path). The ``random`` strategy is intentionally *not* handled
    here so callers keep their exact historical ``torch.random_split`` path.
    """
    strategy = str(data_cfg.get("val_split_strategy", "random")).lower()
    val_split = float(data_cfg["val_split"])
    if strategy == "stratified_gender":
        return stratified_gender_split(
            df,
            val_split=val_split,
            seed=seed,
            val_female_ratio=data_cfg.get("val_female_ratio", 0.5),
        )
    raise ValueError(
        f"make_val_indices does not handle val_split_strategy={strategy!r}; "
        "the 'random' strategy is handled by the caller."
    )
