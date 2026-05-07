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
import json
import os
import pickle
import re
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

def parse_exp_name(exp_name: str) -> dict:
    """Parse an exp_name segment into a hyperparameter dict.

    Format: {alg}[__{key}{value}...}__{reward}
    e.g. ppo__g1_0__n256__a0_95__c0_95__sparse
         grpo__sparse

    Tolerates a stray leading underscore in the value (c_0_95 == c0_95).
    """
    tokens = exp_name.split("__")
    params: dict = {"alg": tokens[0], "reward": tokens[-1]}
    for tok in tokens[1:-1]:
        m = re.match(r'^([a-z]+)_?([\d].*)$', tok)
        if m:
            key, val_str = m.group(1), m.group(2)
            normed = val_str.replace("_", ".")
            try:
                params[key] = float(normed) if "." in normed else int(normed)
            except ValueError:
                params[key] = val_str
    return params


def find_runs(env_id: str, params: dict, n_seeds: int = N_SEEDS) -> list[str]:
    """Return run dirs whose hyperparameters match *params*, one per seed.

    Reads config.json (written by wandb_fetch_runs.py) when present; otherwise
    falls back to parsing hyperparameters from the directory name.
    """
    dirs = []
    for seed in range(1, n_seeds + 1):
        candidates = sorted(
            glob.glob(os.path.join(RUNS_DIR, f"{env_id}__*__{seed}")) +
            glob.glob(os.path.join(RUNS_DIR, f"{env_id}__*__{seed}__*"))
        )
        matched = None
        for path in candidates:
            config_path = os.path.join(path, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path) as f:
                        run_params = json.load(f)
                except Exception:
                    run_params = {}
            else:
                name = os.path.basename(path)
                parts = name.split("__")
                if parts[-1] == str(seed):
                    exp_tokens = parts[1:-1]
                elif len(parts) >= 3 and parts[-2] == str(seed):
                    exp_tokens = parts[1:-2]
                else:
                    continue
                run_params = parse_exp_name("__".join(exp_tokens))
            if all(run_params.get(k) == v for k, v in params.items()):
                matched = path
                break
        if matched:
            dirs.append(matched)
        else:
            print(f"  MISSING: {env_id} seed={seed} {params}")
    return dirs


# ── Aggregation helpers ────────────────────────────────────────────────────────

def _align(curves):
    if not curves:
        return None, None, None
    x_max    = max(s[-1] for s, _ in curves)
    x_common = np.linspace(0, x_max, N_INTERP)

    # Interpolate each seed, marking points beyond its range as NaN.
    mat = np.full((len(curves), N_INTERP), np.nan)
    for i, (s, v) in enumerate(curves):
        mask = x_common <= s[-1]
        mat[i, mask] = np.interp(x_common[mask], s, gaussian_filter1d(v, sigma=SMOOTH_SIGMA))

    n_valid = np.sum(~np.isnan(mat), axis=0)
    # Keep points where at least 1 seed has data.
    keep = n_valid >= 1
    x_common = x_common[keep]
    mat = mat[:, keep]
    n_valid = n_valid[keep]

    mean = np.nanmean(mat, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        std = np.nanstd(mat, axis=0, ddof=1)
    ci95 = np.where(n_valid > 1, 1.96 * std / np.sqrt(n_valid), 0.0)
    return x_common, mean, ci95


def _final_value(curves, last_frac: float = 0.2) -> float:
    """Mean return over the last `last_frac` of each seed's data, averaged across seeds.

    Computed per-seed before averaging so every seed contributes equally,
    regardless of how many steps past MAX_STEPS each one ran.
    """
    seed_finals = []
    for s, v in curves:
        threshold = s[-1] * (1 - last_frac)
        mask = s >= threshold
        if mask.any():
            seed_finals.append(float(np.mean(v[mask])))
    return float(np.mean(seed_finals)) if seed_finals else float("nan")


def load_condition(env_id: str, params: dict,
                   metric: str = RETURN_METRIC,
                   n_seeds: int = N_SEEDS):
    curves = []
    for run_dir in find_runs(env_id, params, n_seeds):
        s, v = _load_scalar(run_dir, metric)
        if s is not None and len(s) > 3:
            curves.append((s, v))
    return curves


def plot_condition(ax, env_id: str, params: dict, label: str, color,
                   linestyle: str = "-",
                   metric: str = RETURN_METRIC,
                   n_seeds: int = N_SEEDS):
    curves = load_condition(env_id, params, metric, n_seeds)
    if not curves:
        print(f"  WARNING: no data for {env_id} {params} {metric}")
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
    P_GRPO   = {"alg": "grpo", "reward": "sparse"}
    P_DENSE  = {"alg": "ppo", "g": 0.999, "n": 256, "a": 0.95, "c": 0.95, "reward": "dense"}
    P_SPARSE = {"alg": "ppo", "g": 0.999, "n": 256, "a": 0.95, "c": 0.95, "reward": "sparse"}
    conditions = [
        (P_GRPO,   "GRPO",       C_GRPO,       "-"),
        (P_DENSE,  "PPO dense",  C_PPO_DENSE,  "-"),
        (P_SPARSE, "PPO sparse", C_PPO_SPARSE, "--"),
    ]

    fig, axes = plt.subplots(1, len(envs), figsize=(4.5 * len(envs), 3.5),
                             sharey=False)
    for ax, env in zip(axes, envs):
        aligned = {}
        for params, label, color, ls in conditions:
            plot_condition(ax, env, params, label, color, ls)
            curves = load_condition(env, params)
            if curves:
                aligned[id(params)] = _align(curves)

        # Arrow annotating the dense→sparse gap where both have data
        d = aligned.get(id(P_DENSE))
        s = aligned.get(id(P_SPARSE))
        if d and s and d[0] is not None and s[0] is not None:
            # Place arrow near the end of the shorter curve
            x_end = min(d[0][-1], s[0][-1])
            x_ann = x_end / 1e6 * 0.92
            y_dense  = float(np.interp(x_ann, d[0] / 1e6, d[1]))
            y_sparse = float(np.interp(x_ann, s[0] / 1e6, s[1]))
            if y_dense != y_sparse:
                ax.annotate(
                    "", xy=(x_ann, y_sparse), xytext=(x_ann, y_dense),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.2),
                )
                mid = (y_dense + y_sparse) / 2
                gap = abs(y_dense - y_sparse)
                ax.text(x_ann * 0.96, mid, f"Δ{gap:.0f}",
                        va="center", ha="right", fontsize=7, color="black")

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
    gammas = [(0.9, "γ=0.9"), (0.95, "γ=0.95"), (0.99, "γ=0.99"), (0.999, "γ=0.999"), (1.0, "γ=1")]

    fig, axes = plt.subplots(1, len(envs), figsize=(4.5 * len(envs), 3.5),
                             sharey=False)
    for ax, env in zip(axes, envs):
        for gamma, glabel in gammas:
            gtag   = str(gamma).replace(".", "_")
            params = {"alg": "ppo", "g": gamma, "n": 256, "a": 0.95, "c": 0.95, "reward": "sparse"}
            color  = GAMMA_COLORS[gtag]
            plot_condition(ax, env, params, glabel, color)
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
    P_GRPO = {"alg": "grpo", "reward": "sparse"}
    P_VF   = {"alg": "ppo", "g": 1.0, "n": 0, "a": 1.0, "c": 1.0, "reward": "sparse"}
    conditions = [
        (P_GRPO, "GRPO",                      C_GRPO,  "-"),
        (P_VF,   "PPO GAE λ=1 (VF baseline)", C_MC_VF, "-"),
    ]

    # Main: episodic return
    fig_ret, axes_ret = plt.subplots(1, len(envs), figsize=(4.5 * len(envs), 3.5))
    for ax, env in zip(axes_ret, envs):
        for params, label, color, ls in conditions:
            plot_condition(ax, env, params, label, color, ls)
        _finish_ax(ax, title=env, legend=(ax is axes_ret[-1]))
    axes_ret[0].set_ylabel("Episodic return (sparse)", fontsize=9)
    for ax in axes_ret[1:]:
        ax.set_ylabel("")
    _shared_legend(fig_ret, axes_ret[-1], ncol=len(conditions))
    fig_ret.tight_layout()
    _save(fig_ret, "fig3_baselines")

    # Diagnostic: explained variance for VF condition, all envs
    print("  Fig 3 (EV diagnostic)")
    ev_metrics = [
        ("losses/explained_variance",    "GAE EV (λ-bootstrap target)", C_MC_VF, "-"),
        ("losses/mc_explained_variance", "MC EV (true returns)",         C_MC_VF, "--"),
    ]
    fig_ev, axes_ev = plt.subplots(1, len(envs), figsize=(4.5 * len(envs), 3.2),
                                   sharey=False)
    for ax, env in zip(axes_ev, envs):
        for metric, label, color, ls in ev_metrics:
            curves = load_condition(env, P_VF, metric=metric)
            if curves:
                x, mean, ci = _align(curves)
                ax.plot(x / 1e6, mean, color=color, lw=LINEWIDTH, ls=ls, label=label)
                ax.fill_between(x / 1e6, mean - ci, mean + ci, color=color, alpha=CI_ALPHA)
        _finish_ax(ax, title=env, ylabel="Explained variance",
                   legend=(ax is axes_ev[-1]))
    axes_ev[0].set_ylabel("Explained variance", fontsize=9)
    for ax in axes_ev[1:]:
        ax.set_ylabel("")
    _shared_legend(fig_ev, axes_ev[-1], ncol=len(ev_metrics))
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
            params = {"alg": "ppo", "g": 1.0, "n": 256, "a": la, "c": lc, "reward": "sparse"}
            curves = load_condition(env, params)
            if curves:
                grid[i, j] = _final_value(curves)

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
        params = {"alg": "ppo", "g": 1.0, "n": 256, "a": la, "c": 0.0, "reward": "sparse"}
        plot_condition(ax, env, params, f"λ_a={la}", colors_la[i])
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

    # GRPO reference
    plot_condition(ax, env, {"alg": "grpo", "reward": "sparse"},
                   "GRPO (episodic, no VF)", C_GRPO, linestyle="--")

    # PPO H sweep (default GAE λ=0.95)
    for H in H_VALS:
        params = {"alg": "ppo", "g": 1.0, "n": H, "a": 0.95, "c": 0.95, "reward": "sparse"}
        plot_condition(ax, env, params, f"PPO H={H}", H_COLORS[H])

    _finish_ax(ax, title=f"{env} — subtrajectory learning (sparse, γ=1)",
               legend_outside=False)
    fig.tight_layout()
    _save(fig, "fig5_subtrajectory_n_sweep")


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
