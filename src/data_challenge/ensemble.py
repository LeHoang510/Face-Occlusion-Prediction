"""Ensemble of multiple prediction CSVs.

Two modes:

  Test mode (default):
      python -m data_challenge.ensemble \
          outputs/run_A/test_predictions.csv \
          outputs/run_B/test_predictions.csv \
          --weights 0.6,0.4 \
          -o outputs/ensemble_test_predictions.csv

      Reads N test_predictions.csv (cols: filename, FaceOcclusion, gender) and
      writes the weighted-average prediction in the same schema (gender="x").

  Val mode (--val):
      python -m data_challenge.ensemble \
          outputs/run_A/val_predictions.csv \
          outputs/run_B/val_predictions.csv \
          --val

      Reads N val_predictions.csv (cols: filename, GT, FaceOcclusion, gender),
      reports the individual scores AND the ensemble score on the validation
      split. With exactly 2 models, also sweeps the weight w in [0, 1] in
      `--n-grid` steps and reports the best.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data_challenge.utils.metrics import compute_score


NAME_COL = "filename"
PRED_COL = "FaceOcclusion"


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if NAME_COL not in df.columns:
        raise ValueError(f"{path}: missing '{NAME_COL}' column")
    if PRED_COL not in df.columns:
        raise ValueError(f"{path}: missing '{PRED_COL}' column")
    return df


def _normalize_weights(weights: list[float], n: int) -> np.ndarray:
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError(f"Got {n} inputs but {len(weights)} weights")
    arr = np.array(weights, dtype=np.float64)
    if arr.sum() <= 0:
        raise ValueError(f"Weights must sum > 0, got {arr.tolist()}")
    return arr / arr.sum()


def _check_aligned(dfs: list[pd.DataFrame], paths: list[Path]) -> None:
    base = dfs[0][NAME_COL].values
    for i, df in enumerate(dfs[1:], start=1):
        if not (df[NAME_COL].values == base).all():
            raise ValueError(f"filename column of {paths[i]} doesn't match {paths[0]}")


def _stack(dfs: list[pd.DataFrame]) -> np.ndarray:
    return np.stack([df[PRED_COL].values.astype(np.float64) for df in dfs], axis=0)


def _weighted_avg(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.clip(stack.T @ weights, 0.0, 1.0)


def ensemble_test(paths: list[Path], weights: list[float] | None, output: Path | None) -> None:
    dfs = [_read(p) for p in paths]
    _check_aligned(dfs, paths)
    w = _normalize_weights(weights, len(dfs))
    avg = _weighted_avg(_stack(dfs), w)

    out_df = pd.DataFrame({
        NAME_COL: dfs[0][NAME_COL].values,
        PRED_COL: avg,
        "gender": "x",
    })

    print(f"[ensemble] {len(paths)} models, weights={w.round(4).tolist()}")
    if output is None:
        out_df.to_csv(sys.stdout, index=False)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output, index=False)
        print(f"[ensemble] wrote: {output}")


def ensemble_val(
    paths: list[Path],
    weights: list[float] | None,
    output: Path | None,
    n_grid: int,
) -> None:
    dfs = [_read(p) for p in paths]
    for p, df in zip(paths, dfs):
        if not {"GT", "gender"} <= set(df.columns):
            raise ValueError(f"{p}: --val mode needs 'GT' and 'gender' columns "
                             "(produced by `eval.py --save-val-preds`)")
    _check_aligned(dfs, paths)

    gt = dfs[0]["GT"].values.astype(np.float64)
    gender = dfs[0]["gender"].values.astype(np.float64)
    stack = _stack(dfs)

    # Per-model scores
    print("[ensemble] per-model val scores:")
    for path, df in zip(paths, dfs):
        s, ef, em = compute_score(df[PRED_COL].values.astype(np.float64), gt, gender)
        print(f"           {path.name:<60} score={s:.5f}  err_F={ef:.5f}  err_M={em:.5f}")

    # Requested-weights ensemble
    w = _normalize_weights(weights, len(dfs))
    avg = _weighted_avg(stack, w)
    s, ef, em = compute_score(avg, gt, gender)
    print(f"[ensemble] requested-weights ensemble (w={w.round(4).tolist()})")
    print(f"           score={s:.5f}  err_F={ef:.5f}  err_M={em:.5f}")

    # Grid search for N=2
    if len(dfs) == 2 and n_grid > 1:
        ws = np.linspace(0.0, 1.0, n_grid)
        best_w, best_score = 0.5, float("inf")
        for w0 in ws:
            avg_g = np.clip(w0 * stack[0] + (1 - w0) * stack[1], 0.0, 1.0)
            s_g, _, _ = compute_score(avg_g, gt, gender)
            if s_g < best_score:
                best_score = s_g
                best_w = float(w0)
        print(f"[ensemble] best-w-search (N=2, grid={n_grid}): w0={best_w:.3f}  score={best_score:.5f}")
        # Optionally write the best-grid ensemble for direct test use
        if output is not None:
            avg_best = np.clip(best_w * stack[0] + (1 - best_w) * stack[1], 0.0, 1.0)
            out_df = pd.DataFrame({
                NAME_COL: dfs[0][NAME_COL].values,
                "GT": gt,
                PRED_COL: avg_best,
                "gender": gender,
            })
            output.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(output, index=False)
            print(f"[ensemble] wrote best-w val predictions to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble multiple prediction CSVs")
    parser.add_argument("paths", nargs="+", help="CSVs to ensemble (>=2)")
    parser.add_argument("--weights", help="comma-separated weights, same order as paths (default: uniform)")
    parser.add_argument("--output", "-o", help="output CSV path (default: stdout for test mode)")
    parser.add_argument(
        "--val",
        action="store_true",
        help="treat inputs as val_predictions.csv (with GT + gender) and report scores",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=21,
        help="(val mode, N=2 only) number of grid points for the weight sweep",
    )
    args = parser.parse_args()

    if len(args.paths) < 2:
        parser.error("need at least 2 prediction CSVs to ensemble")

    paths = [Path(p) for p in args.paths]
    weights = [float(w) for w in args.weights.split(",")] if args.weights else None
    output = Path(args.output) if args.output else None

    if args.val:
        ensemble_val(paths, weights, output, args.n_grid)
    else:
        ensemble_test(paths, weights, output)


if __name__ == "__main__":
    main()
