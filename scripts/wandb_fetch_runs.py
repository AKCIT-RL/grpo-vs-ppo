#!/usr/bin/env python3
"""Print the display name of every finished run in the project, one per line.

Used by experiment scripts to skip runs already completed in wandb:
  wandb_runs=$(mktemp)
  .venv/bin/python scripts/wandb_fetch_runs.py --sync > "$wandb_runs" 2>/dev/null || true
  ...
  grep -qxF "${ENV}__${exp_name}__${seed}" "$wandb_runs" && echo "skip" && continue

With --sync: download tfevents for finished and running runs. Skips runs
  where local data is already up to date (tracked via .sync_step). Finished
  runs get a DONE marker; running runs do not. Falls back to rebuilding from
  W&B history when no tfevents file was uploaded. Skips runs with a LOCK file.
With --clean: delete local dirs and W&B entries for crashed runs with fewer
  than CLEAN_STEP_THRESHOLD steps; sync crashed runs above the threshold.
With --rebuild: reconstruct tfevents from W&B history for all runs (finished,
  running, crashed). Skips runs that already have DONE or are up to date.
With --update: scan all wandb runs and refresh every local copy with the
  latest results. Skips dirs with a LOCK file and no DONE (in-progress locally).
With --workers N: parallelise sync/update/rebuild across N threads (default 8).
With --mark-stale: write DONE to local run dirs whose tfevents have not changed
  in the last hour (configurable with --stale-hours). Purely local, no W&B API.
"""
import argparse
import json
import pathlib
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import wandb

CLEAN_STEP_THRESHOLD = 600_000
DEFAULT_WORKERS = 8


# ── Hyperparameter config ──────────────────────────────────────────────────────

def _extract_params(run: "wandb.apis.public.Run") -> dict:
    """Map a W&B run config to the hyperparameter dict used by plot_paper.py."""
    cfg = run.config
    alg    = "grpo" if cfg.get("grpo", False) else "ppo"
    g      = float(cfg.get("gamma", 0.99))
    n      = int(cfg.get("num_steps", 2048))
    a      = float(cfg.get("gae_lambda_actor", cfg.get("gae_lambda", 0.95)))
    c      = float(cfg.get("gae_lambda_critic", cfg.get("gae_lambda", 0.95)))
    reward = "sparse" if cfg.get("sparse", False) else "dense"
    return {"alg": alg, "g": g, "n": n, "a": a, "c": c, "reward": reward}


def _save_config(run: "wandb.apis.public.Run", run_dir: pathlib.Path) -> None:
    """Write config.json into run_dir so find_runs can match by hyperparameters."""
    params = _extract_params(run)
    (run_dir / "config.json").write_text(json.dumps(params, indent=2) + "\n")


# ── Step tracking (skip-if-up-to-date) ────────────────────────────────────────

def _remote_step(run: "wandb.apis.public.Run") -> int:
    return int(run.summary.get("global_step", run.summary.get("_step", 0)) or 0)


def _local_step(run_dir: pathlib.Path) -> int:
    step_file = run_dir / ".sync_step"
    try:
        return int(step_file.read_text().strip()) if step_file.exists() else -1
    except (ValueError, OSError):
        return -1


def _save_step(run_dir: pathlib.Path, step: int) -> None:
    (run_dir / ".sync_step").write_text(str(step) + "\n")


def _is_up_to_date(run: "wandb.apis.public.Run", run_dir: pathlib.Path) -> bool:
    """True if local .sync_step is at least as far as the remote run."""
    remote = _remote_step(run)
    return remote > 0 and remote <= _local_step(run_dir)


# ── Download helpers ───────────────────────────────────────────────────────────

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
    if _is_up_to_date(run, run_dir):
        return

    rows = list(run.scan_history())
    if not rows:
        print(f"  [warn] no history for {name}", file=sys.stderr)
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    _save_config(run, run_dir)

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

    _save_step(run_dir, _remote_step(run))
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

    if (run_dir / "LOCK").exists():
        return
    if (run_dir / "DONE").exists():
        return
    if _is_up_to_date(run, run_dir):
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    _save_config(run, run_dir)
    downloaded = _download_tfevents(run, run_dir, replace=not write_done)

    if downloaded == 0:
        print(f"  no tfevents for {name}, rebuilding from history", file=sys.stderr)
        rebuild_from_history(run, runs_root, write_done=write_done)
        return

    _save_step(run_dir, _remote_step(run))
    if write_done:
        (run_dir / "DONE").write_text(str(run.summary.get("_timestamp", "")) + "\n")
    print(f"  synced {name} (done={write_done})", file=sys.stderr)


def update_run(
    run: "wandb.apis.public.Run",
    runs_root: pathlib.Path,
) -> None:
    """Force-refresh a local run from wandb. Skip if in-progress locally or up to date."""
    name = run.display_name
    run_dir = runs_root / name

    # In-progress locally: has LOCK but no DONE.
    if (run_dir / "LOCK").exists() and not (run_dir / "DONE").exists():
        return

    # Stale LOCK from a finished run; remove it.
    if (run_dir / "LOCK").exists() and (run_dir / "DONE").exists():
        (run_dir / "LOCK").unlink()

    if _is_up_to_date(run, run_dir):
        return

    write_done = run.state in ("finished", "crashed")

    run_dir.mkdir(parents=True, exist_ok=True)
    _save_config(run, run_dir)
    downloaded = _download_tfevents(run, run_dir, replace=True)

    if downloaded == 0:
        print(f"  no tfevents for {name}, rebuilding from history", file=sys.stderr)
        # Temporarily remove DONE so rebuild_from_history proceeds.
        done_path = run_dir / "DONE"
        if done_path.exists():
            done_path.unlink()
        rebuild_from_history(run, runs_root, write_done=write_done)
        return

    _save_step(run_dir, _remote_step(run))
    if write_done:
        (run_dir / "DONE").write_text(str(run.summary.get("_timestamp", "")) + "\n")
    print(f"  updated {name} (done={write_done})", file=sys.stderr)


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


# ── Stale detection (local, no W&B API) ───────────────────────────────────────

def mark_stale(runs_root: pathlib.Path, max_age_hours: float = 1.0) -> None:
    """Write DONE to local run dirs whose tfevents haven't changed recently.

    Skips dirs that already have DONE. Checks only tfevents file mtimes so
    that bookkeeping files (.sync_step, config.json, LOCK) don't interfere.
    """
    max_age = max_age_hours * 3600
    now = time.time()
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if (run_dir / "DONE").exists():
            continue
        tfevents = list(run_dir.glob("events.out.tfevents.*"))
        if not tfevents:
            continue
        last_change = max(f.stat().st_mtime for f in tfevents)
        age_hours = (now - last_change) / 3600
        if now - last_change > max_age:
            (run_dir / "DONE").write_text("stale\n")
            print(f"  marked stale: {run_dir.name} (last change {age_hours:.1f}h ago)",
                  file=sys.stderr)


# ── Parallel dispatch ──────────────────────────────────────────────────────────

def _run_parallel(func, runs: list, workers: int) -> None:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(func, run): run.display_name for run in runs}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                print(f"  ERROR {futures[fut]}: {exc}", file=sys.stderr)


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
        "--update",
        action="store_true",
        help="Scan all wandb runs and refresh every local copy with latest results.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List finished and running run names (no downloads).",
    )
    parser.add_argument(
        "--check",
        metavar="NAME",
        help="Exit 0 if a run with this display name is running or finished in wandb, else exit 1.",
    )
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel download threads (default {DEFAULT_WORKERS}).")
    parser.add_argument("--mark-stale", action="store_true",
                        help="Write DONE to local run dirs whose tfevents haven't changed recently.")
    parser.add_argument("--stale-hours", type=float, default=1.0,
                        help="Inactivity threshold for --mark-stale in hours (default 1).")
    args = parser.parse_args()

    runs_root = pathlib.Path(args.runs_root)

    if args.mark_stale:
        mark_stale(runs_root, max_age_hours=args.stale_hours)
        return

    api = wandb.Api()

    if args.check:
        matching = api.runs(
            f"{args.entity}/{args.project}",
            filters={"display_name": {"$eq": args.check}, "state": {"$in": ["running", "finished"]}},
            per_page=1,
        )
        sys.exit(0 if list(matching) else 1)

    if args.clean:
        crashed = list(api.runs(
            f"{args.entity}/{args.project}",
            filters={"state": "crashed"},
            per_page=1000,
        ))
        _run_parallel(lambda r: clean_crashed(r, runs_root), crashed, args.workers)

    if args.update:
        all_runs = list(api.runs(
            f"{args.entity}/{args.project}",
            filters={"state": {"$in": ["finished", "running", "crashed"]}},
            per_page=1000,
        ))
        _run_parallel(lambda r: update_run(r, runs_root), all_runs, args.workers)
        return

    if args.rebuild:
        for state, done in (("finished", True), ("running", False), ("crashed", True)):
            runs = list(api.runs(
                f"{args.entity}/{args.project}",
                filters={"state": state},
                per_page=1000,
            ))
            _run_parallel(lambda r, d=done: rebuild_from_history(r, runs_root, write_done=d), runs, args.workers)
        return

    if args.list:
        for run in api.runs(
            f"{args.entity}/{args.project}",
            filters={"state": {"$in": ["finished", "running"]}},
            per_page=1000,
        ):
            print(run.display_name)
        return

    if args.sync:
        running = list(api.runs(
            f"{args.entity}/{args.project}",
            filters={"state": "running"},
            per_page=1000,
        ))
        _run_parallel(lambda r: sync_run(r, runs_root, write_done=False), running, args.workers)

    finished = list(api.runs(
        f"{args.entity}/{args.project}",
        filters={"state": "finished"},
        per_page=1000,
    ))
    if args.sync:
        _run_parallel(lambda r: sync_run(r, runs_root, write_done=True), finished, args.workers)
    for run in finished:
        print(run.display_name)


if __name__ == "__main__":
    main()
