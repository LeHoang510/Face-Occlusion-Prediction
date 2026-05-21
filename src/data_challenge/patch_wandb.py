"""Patch missing metrics into an existing wandb run.

Fetches val/err_female and val/err_male from the run history,
computes the diff, and logs them back at each original step.

Usage:
    python -m data_challenge.patch_wandb --run-id <your_run_id>
    python -m data_challenge.patch_wandb --run-id <your_run_id> --project data-challenge-occlusion
"""

import argparse

import wandb


def patch(run_id: str, project: str, entity: str | None = None):
    api = wandb.Api()

    path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    run = api.run(path)

    history = run.history(keys=["epoch", "val/err_female", "val/err_male"], pandas=True)
    missing = history[["val/err_female", "val/err_male"]].isna().any(axis=1)
    if missing.all():
        print("No val/err_female or val/err_male found in this run's history.")
        return

    rows = history.dropna(subset=["val/err_female", "val/err_male"])
    print(f"Found {len(rows)} steps to patch.")

    resumed = wandb.init(project=project, entity=entity, id=run_id, resume="must")

    # Use a fresh x-axis key ("patch_epoch") that doesn't exist in the run yet,
    # so we don't append to the existing "epoch" series and cause it to hit 40.
    resumed.define_metric("patch_epoch")
    resumed.define_metric("val/err_diff", step_metric="patch_epoch")
    resumed.define_metric("val/err_diff_abs", step_metric="patch_epoch")

    for _, row in rows.iterrows():
        err_f = row["val/err_female"]
        err_m = row["val/err_male"]
        epoch = int(row.get("epoch", row.get("_step", 0)))
        resumed.log({
            "patch_epoch": epoch,
            "val/err_diff": err_f - err_m,
            "val/err_diff_abs": abs(err_f - err_m),
        })
        print(f"  epoch {epoch:>3} | err_diff={err_f - err_m:+.5f} | err_diff_abs={abs(err_f - err_m):.5f}")

    resumed.finish()
    print("Done. Check your wandb run for the new metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch err_diff metrics into an existing wandb run")
    parser.add_argument("--run-id", required=True, help="wandb run ID (the short hash, e.g. 2x3k9abc)")
    parser.add_argument("--project", default="data-challenge-occlusion")
    parser.add_argument("--entity", default=None, help="wandb username or org (optional)")
    args = parser.parse_args()
    patch(args.run_id, project=args.project, entity=args.entity)
