#!/usr/bin/env python3
"""Check experiment status across all five figures."""

import os
from collections import defaultdict
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "runs"

ENVS = ["Humanoid-v4", "Hopper-v4", "Walker2d-v4"]
SEEDS = range(1, 6)


def run_name(env, exp, seed):
    return f"{env}__{exp}__{seed}"


def run_status(name):
    d = RUNS_DIR / name
    if not d.exists():
        return "missing"
    if (d / "DONE").exists():
        return "done"
    if (d / "LOCK").exists():
        return "locked"
    # Directory exists but no DONE/LOCK — check for tfevents
    tfevents = list(d.glob("events.out.tfevents.*"))
    if tfevents:
        return "has_data"
    return "empty_dir"


def build_fig1():
    """Fig 1: Dense vs sparse (3 conditions x 3 envs x 5 seeds = 45)."""
    conditions = [
        "grpo__sparse",
        "ppo__g0_999__n256__a0_95__c_0_95__dense",
        "ppo__g0_999__n256__a0_95__c_0_95__sparse",
    ]
    runs = []
    for seed in SEEDS:
        for env in ENVS:
            for cond in conditions:
                runs.append(run_name(env, cond, seed))
    return runs


def build_fig2():
    """Fig 2: Gamma sweep (5 gammas x 3 envs x 5 seeds = 75)."""
    gammas = ["0_9", "0_95", "0_99", "0_999", "1_0"]
    runs = []
    for seed in SEEDS:
        for env in ENVS:
            for g in gammas:
                exp = f"ppo__g{g}__n256__a0_95__c_0_95__sparse"
                runs.append(run_name(env, exp, seed))
    return runs


def build_fig3():
    """Fig 3: Value function baselines (2 conditions x 3 envs x 5 seeds = 30)."""
    conditions = [
        "grpo__sparse",
        "ppo__g1_0__n0__a1_0__c1_0__sparse",
    ]
    runs = []
    for seed in SEEDS:
        for env in ENVS:
            for cond in conditions:
                runs.append(run_name(env, cond, seed))
    return runs


def build_fig4():
    """Fig 4: Decoupled lambda grid (4x4 x 5 seeds = 80, Humanoid only)."""
    lambdas = ["0_0", "0_5", "0_95", "1_0"]
    env = "Humanoid-v4"
    runs = []
    for seed in SEEDS:
        for la in lambdas:
            for lc in lambdas:
                exp = f"ppo__g1_0__n256__a{la}__c{lc}__sparse"
                runs.append(run_name(env, exp, seed))
    return runs


def build_fig5():
    """Fig 5: Subtrajectory N sweep (1 GRPO + 9 N values) x 5 seeds = 50, Humanoid only."""
    env = "Humanoid-v4"
    n_values = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    runs = []
    for seed in SEEDS:
        runs.append(run_name(env, "grpo__sparse", seed))
        for n in n_values:
            exp = f"ppo__g1_0__n{n}__a0_95__c0_95__sparse"
            runs.append(run_name(env, exp, seed))
    return runs


def print_figure(label, description, runs):
    statuses = {name: run_status(name) for name in runs}

    by_status = defaultdict(list)
    for name, st in statuses.items():
        by_status[st].append(name)

    total = len(runs)
    n_done = len(by_status["done"])
    n_data = len(by_status["has_data"])
    n_locked = len(by_status["locked"])
    n_empty = len(by_status["empty_dir"])
    n_missing = len(by_status["missing"])

    finished = n_done + n_data  # has_data means tfevents exist, likely complete

    print(f"\n{'=' * 72}")
    print(f"{label}: {description}")
    print(f"{'=' * 72}")
    print(f"  Total planned:  {total}")
    print(f"  Done (DONE):    {n_done}")
    print(f"  Has data:       {n_data}")
    print(f"  Locked:         {n_locked}")
    print(f"  Empty dir:      {n_empty}")
    print(f"  Missing:        {n_missing}")
    print(f"  Progress:       {finished}/{total} ({100 * finished / total:.0f}%)")

    if n_missing > 0:
        print(f"\n  Missing runs ({n_missing}):")
        # Group by condition (drop env and seed for readability)
        by_cond = defaultdict(list)
        for name in sorted(by_status["missing"]):
            parts = name.split("__")
            env = parts[0]
            seed = parts[-1]
            cond = "__".join(parts[1:-1])
            by_cond[(env, cond)].append(seed)
        for (env, cond), seeds in sorted(by_cond.items()):
            print(f"    {env} / {cond}  seeds: {', '.join(seeds)}")

    if n_locked > 0:
        print(f"\n  Locked runs ({n_locked}):")
        for name in sorted(by_status["locked"]):
            print(f"    {name}")

    if n_empty > 0:
        print(f"\n  Empty dirs ({n_empty}):")
        for name in sorted(by_status["empty_dir"]):
            print(f"    {name}")


def find_extra_runs():
    """Find run directories not claimed by any figure."""
    all_planned = set()
    for builder in [build_fig1, build_fig2, build_fig3, build_fig4, build_fig5]:
        all_planned.update(builder())

    existing = set()
    if RUNS_DIR.exists():
        for d in RUNS_DIR.iterdir():
            if d.is_dir():
                existing.add(d.name)

    extra = sorted(existing - all_planned)
    if extra:
        print(f"\n{'=' * 72}")
        print(f"Extra runs not in any figure ({len(extra)})")
        print(f"{'=' * 72}")
        for name in extra:
            st = run_status(name)
            print(f"  [{st:>8}] {name}")


def main():
    print("Experiment status overview")
    print(f"Runs directory: {RUNS_DIR}")

    figures = [
        ("Fig 1", "Dense vs sparse setup validation (3 cond x 3 env x 5 seeds)", build_fig1),
        ("Fig 2", "Gamma sweep (5 gamma x 3 env x 5 seeds)", build_fig2),
        ("Fig 3", "Value function as baseline (2 cond x 3 env x 5 seeds)", build_fig3),
        ("Fig 4", "Decoupled lambda grid (4x4 x 5 seeds, Humanoid)", build_fig4),
        ("Fig 5", "Subtrajectory N sweep (10 cond x 5 seeds, Humanoid)", build_fig5),
    ]

    # Compute unique runs across all figures
    all_unique = set()
    for _, _, builder in figures:
        all_unique.update(builder())

    for label, desc, builder in figures:
        runs = builder()
        print_figure(label, desc, runs)

    find_extra_runs()

    # Summary
    print(f"\n{'=' * 72}")
    print("Summary")
    print(f"{'=' * 72}")
    total_unique = len(all_unique)
    done_unique = sum(1 for r in all_unique if run_status(r) in ("done", "has_data"))
    missing_unique = sum(1 for r in all_unique if run_status(r) == "missing")
    print(f"  Unique runs across all figures: {total_unique}")
    print(f"  Completed (done + has_data):    {done_unique}")
    print(f"  Missing (no directory):         {missing_unique}")
    print(f"  Overall progress:               {done_unique}/{total_unique} ({100 * done_unique / total_unique:.0f}%)")


if __name__ == "__main__":
    main()
