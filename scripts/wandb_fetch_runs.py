#!/usr/bin/env python3
"""Print the display name of every finished run in the project, one per line.

Used by experiment scripts to skip runs already completed in wandb:
  wandb_runs=$(mktemp)
  .venv/bin/python scripts/wandb_fetch_runs.py --sync > "$wandb_runs" 2>/dev/null || true
  ...
  grep -qxF "${ENV}__${exp_name}__${seed}" "$wandb_runs" && echo "skip" && continue

With --sync: download tfevents for finished and running runs. Finished runs
  get a DONE marker; running runs do not. Falls back to rebuilding from W&B
  history when no tfevents file was uploaded. Skips runs that have a local
  LOCK file (running on this machine).
With --clean: delete local dirs and W&B entries for crashed runs with fewer
  than CLEAN_STEP_THRESHOLD steps; sync crashed runs above the threshold.
With --rebuild: reconstruct tfevents from W&B history for all runs (finished,
  running, crashed). Skips runs that already have DONE.
"""
import argparse
import pathlib
import shutil
import sys

import wandb

CLEAN_STEP_THRESHOLD = 600_000


def _download_tfevents(
    run: "wandb.apis.public.Run",
    run_dir: pathlib.Path,
    replace: bool = False,
) -> int:
    """Download tfevents files into run_dir. Returns count downloaded.

    Flattens W&B's internal directory structure and removes the empty
    subdirectories it creates (e.g. runs/{name}/ inside run_dir).
    """
    downloaded = 0
    for f in run.files():
        if "tfevents" not in f.name:
            continue
        dest = run_dir / pathlib.Path(f.name).name
        if not replace and dest.exists():
            downloaded += 1
            continue
        f.download(root=str(run_dir), replace=replace)
        # wandb recreates its internal path; move the file up and clean dirs.
        downloaded_path = run_dir / f.name
        if downloaded_path.exists() and downloaded_path != dest:
            downloaded_path.rename(dest)
        downloaded += 1

    # Remove empty subdirectories left by wandb's download.
    for d in sorted(run_dir.rglob("*/"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()

    return downloaded


def rebuild_from_history(
    run: "wandb.apis.public.Run",
    runs_root: pathlib.Path,
    write_done: bool = False,
) -> None:
    """Reconstruct a tfevents file from W&B history scalars.

    Works for any run where sync_tensorboard=True was set, regardless of
    whether the raw .tfevents file was uploaded.
    """
    from torch.utils.tensorboard import SummaryWriter

    name = run.display_name
    run_dir = runs_root / name

    if (run_dir / "DONE").exists():
        return

    rows = list(run.scan_history())
    if not rows:
        print(f"  [warn] no history for {name}", file=sys.stderr)
        return

    run_dir.mkdir(parents=True, exist_ok=True)

    # Remove any existing tfevents so we write a single clean file.
    for old in run_dir.glob("events.out.tfevents.*"):
        old.unlink()

    writer = SummaryWriter(log_dir=str(run_dir))
    for row in rows:
        step = row.get("global_step") or row.get("_step")
        if step is None:
            continue
        step = int(step)
        for key, val in row.items():
            if key.startswith("_") or key == "global_step":
                continue
            if not isinstance(val, (int, float)):
                continue
            writer.add_scalar(key, val, global_step=step)
    writer.close()

    if write_done:
        (run_dir / "DONE").write_text(str(run.summary.get("_timestamp", "")) + "\n")
    print(f"  rebuilt {name} ({len(rows)} rows, done={write_done})", file=sys.stderr)


def sync_run(
    run: "wandb.apis.public.Run",
    runs_root: pathlib.Path,
    write_done: bool = True,
) -> None:
    """Download tfevents for a run. Falls back to history rebuild if no file exists."""
    name = run.display_name
    run_dir = runs_root / name

    # Running locally on this machine; don't touch.
    if (run_dir / "LOCK").exists():
        return

    if (run_dir / "DONE").exists():
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    downloaded = _download_tfevents(run, run_dir, replace=not write_done)

    if downloaded == 0:
        print(f"  no tfevents for {name}, rebuilding from history", file=sys.stderr)
        rebuild_from_history(run, runs_root, write_done=write_done)
        return

    if write_done:
        (run_dir / "DONE").write_text(str(run.summary.get("_timestamp", "")) + "\n")
    print(f"  synced {name} (done={write_done})", file=sys.stderr)


def clean_crashed(
    run: "wandb.apis.public.Run",
    runs_root: pathlib.Path,
) -> None:
    """Delete crashed runs below the step threshold; sync those above it."""
    name = run.display_name
    steps = run.summary.get("global_step", run.summary.get("_step", 0)) or 0

    if steps >= CLEAN_STEP_THRESHOLD:
        print(f"  syncing crashed run {name} ({steps:,} steps)", file=sys.stderr)
        sync_run(run, runs_root, write_done=True)
        return

    run_dir = runs_root / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
        print(f"  deleted local dir for crashed run {name} ({steps:,} steps)", file=sys.stderr)
    else:
        print(f"  crashed run has no local dir: {name} ({steps:,} steps)", file=sys.stderr)

    run.delete()
    print(f"  deleted from wandb: {name}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="grpo-vs-ppo")
    parser.add_argument("--entity", default="marcospaulo2-federal-university-of-goi-s")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Download tfevents for finished and running runs.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete crashed runs below the step threshold; sync those above it.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reconstruct tfevents from W&B history for all runs.",
    )
    parser.add_argument(
        "--check",
        metavar="NAME",
        help="Exit 0 if a run with this display name is running or finished in wandb, else exit 1.",
    )
    parser.add_argument("--runs-root", default="runs")
    args = parser.parse_args()

    api = wandb.Api()

    if args.check:
        matching = api.runs(
            f"{args.entity}/{args.project}",
            filters={"display_name": {"$eq": args.check}, "state": {"$in": ["running", "finished"]}},
            per_page=1,
        )
        sys.exit(0 if list(matching) else 1)

    runs_root = pathlib.Path(args.runs_root)

    if args.clean:
        for run in api.runs(
            f"{args.entity}/{args.project}",
            filters={"state": "crashed"},
            per_page=1000,
        ):
            clean_crashed(run, runs_root)

    if args.rebuild:
        for state, done in (("finished", True), ("running", False), ("crashed", True)):
            for run in api.runs(
                f"{args.entity}/{args.project}",
                filters={"state": state},
                per_page=1000,
            ):
                rebuild_from_history(run, runs_root, write_done=done)
        return

    if args.sync:
        for run in api.runs(
            f"{args.entity}/{args.project}",
            filters={"state": "running"},
            per_page=1000,
        ):
            sync_run(run, runs_root, write_done=False)

    for run in api.runs(
        f"{args.entity}/{args.project}",
        filters={"state": "finished"},
        per_page=1000,
    ):
        if args.sync:
            sync_run(run, runs_root, write_done=True)
        print(run.display_name)


if __name__ == "__main__":
    main()

