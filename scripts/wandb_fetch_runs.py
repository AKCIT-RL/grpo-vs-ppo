#!/usr/bin/env python3
"""Print the display name of every finished run in the project, one per line.

Used by experiment scripts to skip runs already completed in wandb:
  wandb_runs=$(mktemp)
  .venv/bin/python scripts/wandb_fetch_runs.py --sync > "$wandb_runs" 2>/dev/null || true
  ...
  grep -qxF "${ENV}__${exp_name}__${seed}" "$wandb_runs" && echo "skip" && continue

With --sync the script also:
  Finished runs -- create runs/{run_name}/ if needed, download tfevents, write DONE.
  Crashed runs  -- delete runs/{run_name}/ and locks/{run_name}.lock so the
                   experiment scripts will re-queue them.
"""
import argparse
import pathlib
import shutil
import sys

import wandb


def sync_finished(run: "wandb.apis.public.Run", runs_root: pathlib.Path) -> None:
    name = run.display_name
    run_dir = runs_root / name

    if (run_dir / "DONE").exists():
        return

    run_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for f in run.files():
        if "tfevents" not in f.name:
            continue
        dest = run_dir / pathlib.Path(f.name).name
        if not dest.exists():
            f.download(root=str(run_dir), replace=False)
            # wandb may recreate subdirectory structure; flatten to run_dir.
            downloaded_path = run_dir / f.name
            if downloaded_path.exists() and downloaded_path != dest:
                downloaded_path.rename(dest)
        downloaded += 1

    if downloaded == 0:
        print(f"  [warn] no tfevents found for {name}", file=sys.stderr)

    (run_dir / "DONE").write_text(str(run.summary.get("_timestamp", "")) + "\n")
    print(f"  synced {name}", file=sys.stderr)


def sync_crashed(
    run: "wandb.apis.public.Run",
    runs_root: pathlib.Path,
) -> None:
    name = run.display_name
    run_dir = runs_root / name

    if run_dir.exists():
        shutil.rmtree(run_dir)
        print(f"  deleted crashed run dir {name}", file=sys.stderr)
    else:
        print(f"  crashed run has no local dir: {name}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="grpo-vs-ppo")
    parser.add_argument("--entity", default="marcospaulo2-federal-university-of-goi-s")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Download tfevents + write DONE for finished runs; delete dirs for crashed runs.",
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
        # Check for running or finished — another machine may have started or
        # completed the run after our initial batch fetch.
        matching = api.runs(
            f"{args.entity}/{args.project}",
            filters={"display_name": {"$eq": args.check}, "state": {"$in": ["running", "finished"]}},
            per_page=1,
        )
        sys.exit(0 if list(matching) else 1)

    runs_root = pathlib.Path(args.runs_root)

    api = wandb.Api()

    if args.sync:
        crashed = api.runs(
            f"{args.entity}/{args.project}",
            filters={"state": "crashed"},
            per_page=1000,
        )
        for run in crashed:
            sync_crashed(run, runs_root)

    finished = api.runs(
        f"{args.entity}/{args.project}",
        filters={"state": "finished"},
        per_page=1000,
    )
    for run in finished:
        if args.sync:
            sync_finished(run, runs_root)
        print(run.display_name)


if __name__ == "__main__":
    main()

