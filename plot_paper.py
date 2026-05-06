#!/usr/bin/env python3
"""
Paper figures for the GRPO-vs-PPO NeurIPS 2026 paper.
Reads TensorBoard event files from runs/, outputs PNGs to paper/fig/.

Run dirs have the form:  runs/{env_id}__{exp_name}__{seed}__{timestamp}/
The exp_name is set by --exp-name when launching experiments.

Usage:
    .venv/bin/python plot_paper.py              # all figures
    .venv/bin/python plot_paper.py fig1 fig4    # specific figures
    .venv/bin/python plot_paper.py --list       # list available figures
"""
import argparse
import glob
import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Config ────────────────────────────────────────────────────────────────────

RUNS_DIR   = "runs"
FIG_DIR    = "paper/fig"
CACHE_FILE = os.path.join(RUNS_DIR, ".tb_cache.pkl")
RETURN_METRIC = "charts/episodic_return"
MAX_STEPS  = 1_000_000
SMOOTH_SIGMA = 3
N_INTERP   = 300
CI_ALPHA   = 0.15
LINEWIDTH  = 1.8
N_SEEDS    = 5

os.makedirs(FIG_DIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────

C_GRPO         = "#d62728"   # red    — GRPO reference
C_PPO_DENSE    = "#1f77b4"   # blue   — PPO dense default
C_PPO_SPARSE   = "#ff7f0e"   # orange — PPO sparse (same config, sparse reward)

# Fig 2: gamma sweep — viridis ramp, light→dark as γ increases
_G_CMAP    = plt.cm.viridis
GAMMA_COLORS = {
    "0_9":   _G_CMAP(0.10),
    "0_95":  _G_CMAP(0.35),
    "0_99":  _G_CMAP(0.60),
    "0_999": _G_CMAP(0.78),
    "1_0":   _G_CMAP(0.95),
}

# Fig 3: baselines
C_MC_VF   = "#2ca02c"   # green — PPO MC with VF baseline (GAE λ=1)
C_MC_BM   = "#9467bd"   # purple — PPO MC with batch_mean, no VF

# Fig 4: lambda grid — two families, one per lambda_actor value
_LA_CMAP = plt.cm.Blues
_LC_CMAP = plt.cm.Oranges
LAM_VALS = [0.0, 0.5, 0.95, 1.0]

# Fig 5: H sweep — viridis ramp light→dark as H increases
_H_CMAP = plt.cm.viridis
H_VALS   = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
H_COLORS = {H: _H_CMAP(i / (len(H_VALS) - 1)) for i, H in enumerate(H_VALS)}

# ── Cache ──────────────────────────────────────────────────────────────────────

_cache: dict = {}
_ea_session: dict = {}  # run_dir → (EA, mtime)


def _load_cache():
    global _cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                _cache = pickle.load(f)
            print(f"  Cache: {len(_cache)} entries from {CACHE_FILE}")
        except Exception as e:
            print(f"  WARNING: cache unreadable ({e}), starting fresh")
            _cache = {}


def _save_cache():
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(_cache, f, protocol=4)
    except Exception as e:
        print(f"  WARNING: could not save cache: {e}")


def _run_mtime(run_dir: str) -> float:
    mtime = 0.0
    try:
        for entry in os.scandir(run_dir):
            if entry.is_file():
                mtime = max(mtime, entry.stat().st_mtime)
    except FileNotFoundError:
        pass
    return mtime


def _get_ea(run_dir: str) -> EventAccumulator:
    mtime = _run_mtime(run_dir)
    cached = _ea_session.get(run_dir)
    if cached is not None and cached[1] == mtime:
        return cached[0]
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ea.Reload()
    _ea_session[run_dir] = (ea, mtime)
    return ea


def _load_scalar(run_dir: str, metric: str):
    cache_key = f"{run_dir}:{metric}"
    mtime = _run_mtime(run_dir)
    entry = _cache.get(cache_key)
    if entry is not None and entry["mtime"] == mtime:
        return entry.get("steps"), entry.get("values")
    try:
        ea = _get_ea(run_dir)
        events = ea.Scalars(metric)
    except KeyError:
        _cache[cache_key] = {"mtime": mtime, "steps": None, "values": None}
        return None, None
    steps  = np.array([e.step  for e in events], dtype=float)
    values = np.array([e.value for e in events], dtype=float)
    mask = steps <= MAX_STEPS
    steps, values = steps[mask], values[mask]
    _cache[cache_key] = {"mtime": mtime, "steps": steps, "values": values}
    return steps, values


# ── Run discovery ──────────────────────────────────────────────────────────────

def find_runs(env_id: str, exp_name: str, n_seeds: int = N_SEEDS) -> list[str]:
    """Return sorted list of run dirs matching {env_id}__{exp_name}__{seed}
    for seeds 1..n_seeds.  Checks exact path first, falls back to __* glob
    for legacy dirs with timestamps."""
    dirs = []
    for seed in range(1, n_seeds + 1):
        exact = os.path.join(RUNS_DIR, f"{env_id}__{exp_name}__{seed}")
        if os.path.isdir(exact):
            dirs.append(exact)
            continue
        pattern = os.path.join(RUNS_DIR, f"{env_id}__{exp_name}__{seed}__*")
        matches = sorted(glob.glob(pattern))
        if matches:
            dirs.append(matches[-1])  # most recent
        else:
            print(f"  MISSING: {env_id}__{exp_name}__{seed}")
    return dirs


# ── Aggregation helpers ────────────────────────────────────────────────────────

def _align(curves):
    if not curves:
        return None, None, None
    x_max    = min(s[-1] for s, _ in curves)
    x_common = np.linspace(0, x_max, N_INTERP)
    mat = [np.interp(x_common, s, gaussian_filter1d(v, sigma=SMOOTH_SIGMA))
           for s, v in curves]
    mat  = np.array(mat)
    mean = mat.mean(axis=0)
    ci95 = 1.96 * mat.std(axis=0, ddof=1) / np.sqrt(len(mat))
    return x_common, mean, ci95


def load_condition(env_id: str, exp_name: str,
                   metric: str = RETURN_METRIC,
                   n_seeds: int = N_SEEDS):
    curves = []
    for run_dir in find_runs(env_id, exp_name, n_seeds):
        s, v = _load_scalar(run_dir, metric)
        if s is not None and len(s) > 3:
            curves.append((s, v))
    return curves


def plot_condition(ax, env_id: str, exp_name: str, label: str, color,
                   linestyle: str = "-",
                   metric: str = RETURN_METRIC,
                   n_seeds: int = N_SEEDS):
    curves = load_condition(env_id, exp_name, metric, n_seeds)
    if not curves:
        print(f"  WARNING: no data for {env_id}/{exp_name}/{metric}")
        return None
    x, mean, ci = _align(curves)
    ax.plot(x / 1e6, mean, color=color, lw=LINEWIDTH, ls=linestyle, label=label)
    ax.fill_between(x / 1e6, mean - ci, mean + ci, color=color, alpha=CI_ALPHA)
    return float(mean[-1])


# ── Axes helpers ───────────────────────────────────────────────────────────────

def _finish_ax(ax, title=None, xlabel="Environment steps (×10⁶)",
               ylabel="Episodic return", legend=True, legend_outside=True):
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        if legend_outside:
            ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1),
                      framealpha=0.9, borderaxespad=0)
        else:
            ax.legend(fontsize=7, framealpha=0.9)


def _save(fig, name):
    path = os.path.join(FIG_DIR, name if name.endswith(".png") else name + ".png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _shared_legend(fig, ax, ncol=3):
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.04), handlelength=2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Dense vs sparse vs GRPO
# ═══════════════════════════════════════════════════════════════════════════════

def fig1():
    print("Fig 1: dense vs sparse vs GRPO")
    envs = ["Humanoid-v4", "Hopper-v4", "Walker2d-v4"]
    conditions = [
        ("grpo__sparse",                          "GRPO",        C_GRPO,       "-"),
        ("ppo__g0_999__n256__a0_95__c_0_95__dense",  "PPO dense",   C_PPO_DENSE,  "-"),
        ("ppo__g0_999__n256__a0_95__c_0_95__sparse", "PPO sparse",  C_PPO_SPARSE, "--"),
    ]

    fig, axes = plt.subplots(1, len(envs), figsize=(4.5 * len(envs), 3.5),
                             sharey=False)
    for ax, env in zip(axes, envs):
        for exp_name, label, color, ls in conditions:
            plot_condition(ax, env, exp_name, label, color, ls)
        _finish_ax(ax, title=env, legend=(ax is axes[-1]))
    axes[0].set_ylabel("Episodic return", fontsize=9)
    for ax in axes[1:]:
        ax.set_ylabel("")

    _shared_legend(fig, axes[-1], ncol=len(conditions))
    fig.tight_layout()
    _save(fig, "fig1_dense_vs_sparse")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Gamma sweep
# ═══════════════════════════════════════════════════════════════════════════════

def fig2():
    print("Fig 2: gamma sweep")
    envs = ["Humanoid-v4", "Hopper-v4", "Walker2d-v4"]
    gammas = [("0_9", "γ=0.9"), ("0_95", "γ=0.95"), ("0_99", "γ=0.99"), ("0_999", "γ=0.999"), ("1_0", "γ=1")]

    fig, axes = plt.subplots(1, len(envs), figsize=(4.5 * len(envs), 3.5),
                             sharey=False)
    for ax, env in zip(axes, envs):
        for gtag, glabel in gammas:
            exp_name = f"ppo__g{gtag}__n256__a0_95__c_0_95__sparse"
            color    = GAMMA_COLORS[gtag]
            plot_condition(ax, env, exp_name, glabel, color)
        _finish_ax(ax, title=env, legend=(ax is axes[-1]))
    axes[0].set_ylabel("Episodic return (sparse)", fontsize=9)
    for ax in axes[1:]:
        ax.set_ylabel("")

    _shared_legend(fig, axes[-1], ncol=len(gammas))
    fig.tight_layout()
    _save(fig, "fig2_gamma_sweep")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Role of the VF as a baseline
# ═══════════════════════════════════════════════════════════════════════════════

def fig3():
    print("Fig 3: VF as baseline")
    envs = ["Humanoid-v4", "Hopper-v4", "Walker2d-v4"]
    conditions = [
        ("grpo_sparse",                    "GRPO",                    C_GRPO,  "-"),
        ("ppo__g1_0__n0__a1_0__c1_0__sparse", "PPO GAE λ=1 (VF baseline)", C_MC_VF, "-"),
    ]

    # Main: episodic return
    fig_ret, axes_ret = plt.subplots(1, len(envs), figsize=(4.5 * len(envs), 3.5))
    for ax, env in zip(axes_ret, envs):
        for exp_name, label, color, ls in conditions:
            plot_condition(ax, env, exp_name, label, color, ls)
        _finish_ax(ax, title=env, legend=(ax is axes_ret[-1]))
    axes_ret[0].set_ylabel("Episodic return (sparse)", fontsize=9)
    for ax in axes_ret[1:]:
        ax.set_ylabel("")
    _shared_legend(fig_ret, axes_ret[-1], ncol=len(conditions))
    fig_ret.tight_layout()
    _save(fig_ret, "fig3_baselines")

    # Diagnostic: explained variance for VF condition
    print("  Fig 3 (EV diagnostic)")
    ev_metrics = [
        ("losses/explained_variance",    "GAE EV  (λ-bootstrap target)", C_MC_VF, "-"),
        ("losses/mc_explained_variance", "MC EV   (true returns)",        C_MC_VF, "--"),
    ]
    env = "Humanoid-v4"
    exp_name = "ppo__g1_0__n0__a1_0__c1_0__sparse"
    fig_ev, ax = plt.subplots(figsize=(4.5, 3.2))
    for metric, label, color, ls in ev_metrics:
        curves = load_condition(env, exp_name, metric=metric)
        if curves:
            x, mean, ci = _align(curves)
            ax.plot(x / 1e6, mean, color=color, lw=LINEWIDTH, ls=ls, label=label)
            ax.fill_between(x / 1e6, mean - ci, mean + ci, color=color, alpha=CI_ALPHA)
    _finish_ax(ax, title=f"{env} — VF explained variance",
               ylabel="Explained variance", legend_outside=False)
    fig_ev.tight_layout()
    _save(fig_ev, "fig3_ev_diagnostic")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Decoupled lambda grid
# ═══════════════════════════════════════════════════════════════════════════════

def fig4():
    print("Fig 4: decoupled lambda grid")
    env = "Humanoid-v4"

    # Heatmap: final mean return for each (λ_actor, λ_critic) cell.
    grid = np.full((len(LAM_VALS), len(LAM_VALS)), np.nan)
    for i, la in enumerate(LAM_VALS):
        for j, lc in enumerate(LAM_VALS):
            atag = str(la).replace(".", "_")
            ctag = str(lc).replace(".", "_")
            exp_name = f"ppo__g1_0__n256__a{atag}__c{ctag}__sparse"
            curves = load_condition(env, exp_name)
            if curves:
                x, mean, _ = _align(curves)
                grid[i, j] = mean[-1]

    fig_heat, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(grid, aspect="auto", origin="lower",
                   cmap="viridis", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Final episodic return")
    ticks = [str(v) for v in LAM_VALS]
    ax.set_xticks(range(len(LAM_VALS)))
    ax.set_xticklabels(ticks, fontsize=8)
    ax.set_yticks(range(len(LAM_VALS)))
    ax.set_yticklabels(ticks, fontsize=8)
    ax.set_xlabel("λ_critic", fontsize=9)
    ax.set_ylabel("λ_actor", fontsize=9)
    ax.set_title(f"{env} — final return (sparse, γ=1, H=256)", fontsize=9)
    for i in range(len(LAM_VALS)):
        for j in range(len(LAM_VALS)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if grid[i, j] < np.nanmedian(grid) else "black")
    fig_heat.tight_layout()
    _save(fig_heat, "fig4_lambda_grid_heatmap")

    # Learning curves: fix λ_critic=0 (best TD bootstrap), sweep λ_actor
    fig_lc, ax = plt.subplots(figsize=(5, 3.5))
    n_actor = len(LAM_VALS)
    colors_la = [_LA_CMAP(0.3 + 0.6 * i / (n_actor - 1)) for i in range(n_actor)]
    for i, la in enumerate(LAM_VALS):
        atag = str(la).replace(".", "_")
        exp_name = f"ppo__g1_0__n256__a{atag}__c0_0__sparse"
        plot_condition(ax, env, exp_name, f"λ_a={la}", colors_la[i])
    _finish_ax(ax, title=f"{env} — λ_critic=0, sweep λ_actor (sparse, γ=1, H=256)",
               legend_outside=False)
    fig_lc.tight_layout()
    _save(fig_lc, "fig4_lambda_actor_sweep")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Subtrajectory learning
# ═══════════════════════════════════════════════════════════════════════════════

def fig5():
    print("Fig 5: subtrajectory H sweep")
    env = "Humanoid-v4"

    fig, ax = plt.subplots(figsize=(6, 4))

    # GRPO reference (shared exp_name with fig3)
    plot_condition(ax, env, "grpo_sparse", "GRPO (episodic, no VF)",
                   C_GRPO, linestyle="--")

    # PPO H sweep (default GAE λ=0.95)
    for i, H in enumerate(H_VALS):
        exp_name = f"ppo__g1_0__n{H}__a0_95__c0_95__sparse"
        color = H_COLORS[H]
        plot_condition(ax, env, exp_name, f"PPO H={H}", color)

    _finish_ax(ax, title=f"{env} — subtrajectory learning (sparse, γ=1)",
               legend_outside=False)
    fig.tight_layout()
    _save(fig, "fig5_subtrajectory_n_sweep")

    # Bar chart: final return vs H
    final_returns = {}
    grpo_curves = load_condition(env, "grpo_sparse")
    if grpo_curves:
        _, grpo_mean, _ = _align(grpo_curves)
        final_returns["GRPO"] = grpo_mean[-1]
    for H in H_VALS:
        curves = load_condition(env, f"ppo__g1_0__n{H}__a0_95__c0_95__sparse")
        if curves:
            _, m, _ = _align(curves)
            final_returns[str(H)] = m[-1]

    if final_returns:
        fig_bar, ax_bar = plt.subplots(figsize=(7, 3.5))
        labels = list(final_returns.keys())
        vals   = list(final_returns.values())
        colors = [C_GRPO] + [H_COLORS[H] for H in H_VALS if str(H) in final_returns]
        ax_bar.bar(range(len(labels)), vals, color=colors[:len(labels)])
        ax_bar.set_xticks(range(len(labels)))
        ax_bar.set_xticklabels(labels, fontsize=8)
        ax_bar.set_xlabel("H (rollout steps) / method", fontsize=9)
        ax_bar.set_ylabel("Final episodic return", fontsize=9)
        ax_bar.set_title(f"{env} — final return vs rollout length (sparse, γ=1)", fontsize=9)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        fig_bar.tight_layout()
        _save(fig_bar, "fig5_subtrajectory_final_return")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

FIGURES = {
    "fig1": fig1,
    "fig2": fig2,
    "fig3": fig3,
    "fig4": fig4,
    "fig5": fig5,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("figures", nargs="*", help="Figures to plot (default: all)")
    parser.add_argument("--list", action="store_true", help="List available figures and exit")
    args = parser.parse_args()

    if args.list:
        for name in FIGURES:
            print(name)
        raise SystemExit(0)

    _load_cache()

    to_plot = args.figures if args.figures else list(FIGURES.keys())
    for name in to_plot:
        if name not in FIGURES:
            print(f"WARNING: unknown figure '{name}', skipping")
            continue
        FIGURES[name]()

    _save_cache()
    print("Done.")
