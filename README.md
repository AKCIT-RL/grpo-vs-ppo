# grpo-vs-ppo

## Setup

```bash
uv sync --extra mujoco
```

## Running experiments

Each figure has a corresponding launch script in `scripts/`. All scripts share the same interface:

```
./scripts/experiment-fig<N>-<name>.sh <num_instances> <cur_instance> [--jobs J] [--dry-run]
```

- `num_instances` — total number of nodes splitting the work; the algorithms are CPU-bound, so use the amount of physical CPUs here
- `cur_instance` — 0-indexed ID of the current node
- `--jobs J` — number of parallel training processes per node (default: 1); use around the number of CPU cores you have
- `--dry-run` — print the planned runs without executing anything

The scripts round-robin the full run list across instances by index, so each node gets a non-overlapping subset. Completed runs (those with a `runs/<name>/DONE` marker or already logged to wandb) are automatically skipped, making it safe to restart or add nodes mid-way.

### Single machine

Run everything sequentially on one machine:

```bash
./scripts/experiment-fig1-dense-vs-sparse.sh 1 0
```

Use `--jobs` to saturate CPU cores. A good default is the number of physical cores divided by 2 (each MuJoCo run uses ~2 cores):

```bash
./scripts/experiment-fig1-dense-vs-sparse.sh 1 0 --jobs 8
```

### Multiple machines (recommended)

Split across N machines by assigning each a unique `cur_instance` in `[0, N)`. On a 4-node cluster, run one of these per machine:

```bash
# machine 0
./scripts/experiment-fig1-dense-vs-sparse.sh 4 0 --jobs 8

# machine 1
./scripts/experiment-fig1-dense-vs-sparse.sh 4 1 --jobs 8

# machine 2
./scripts/experiment-fig1-dense-vs-sparse.sh 4 2 --jobs 8

# machine 3
./scripts/experiment-fig1-dense-vs-sparse.sh 4 3 --jobs 8
```

### Running all figures

To reproduce the full paper, launch all five experiment scripts. They share run deduplication, so overlapping conditions (e.g. GRPO sparse runs used in fig1, fig3, and fig5) only train once:

```bash
for fig in fig1-dense-vs-sparse fig2-gamma-sweep fig3-baselines fig4-lambda-grid fig5-subtrajectory; do
  ./scripts/experiment-${fig}.sh 4 $NODE_ID --jobs 8 &
done
wait
```

Set `NODE_ID` to the current machine's index and adjust `--jobs` based on available cores.