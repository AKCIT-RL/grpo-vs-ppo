#!/usr/bin/env bash
# Fig 3 — Role of the value function as a baseline
#
# 3 conditions × 3 envs × 5 seeds = 45 runs.
# grpo_g1_0_sparse runs are shared with fig1 and fig5 — skipped automatically.
#
# Conditions:
#   grpo_g1_0_sparse       GRPO: no VF, group_mean+std, γ=1, sparse (shared with fig1)
#   ppo_mc_vf_g1_0_sparse  GAE λ=1 ≡ MC returns with VF as baseline, γ=1, sparse
#   ppo_mc_bm_g1_0_sparse  MC returns, group_mean (no std scaling), no VF, γ=1, sparse
#                          Isolates VF contribution vs the GRPO variance-reduction trick
#
# Episodic mode (num_steps=0) throughout.  The VF condition logs explained_variance,
# letting us verify the VF is well-fitted even if it offers no performance advantage.
#
# Usage:
#   Single GPU:  ./scripts/experiment-fig3-baselines.sh 1 0
#   With jobs:   ./scripts/experiment-fig3-baselines.sh 1 0 --jobs 9
#   Dry run:      ./scripts/experiment-fig3-baselines.sh --dry-run

set -Eeuo pipefail

num_instances=""
cur_instance=""
jobs_per_instance=1
dry_run=false

while [ "$#" -gt 0 ]; do
  case "$1" in
    --jobs)               jobs_per_instance="$2";   shift 2 ;;
    --dry-run)            dry_run=true;               shift ;;
    *)
      if   [ -z "$num_instances" ]; then num_instances="$1"
      elif [ -z "$cur_instance"  ]; then cur_instance="$1"
      else echo "Error: unknown argument: $1" >&2; exit 1
      fi
      shift ;;
  esac
done

if [ -z "$num_instances" ] || [ -z "$cur_instance" ]; then
  if $dry_run; then
    num_instances=1; cur_instance=0
  else
    echo "Usage: $0 <num_instances> <cur_instance> [--jobs J] [--dry-run]"
    exit 1
  fi
fi

envs=(Humanoid-v4 Hopper-v4 Walker2d-v4)

# GAE with lambda=1 in episode mode is equivalent to MC returns with VF as baseline.
# exp_name tag convention: float values use dot-to-underscore (1.0 -> 1_0)
# grpo_g1_0_sparse is shared with fig1 and fig5 -- skipped automatically.
declare -a CONDITIONS=(
  "grpo_g1_0_sparse --grpo --sparse"
  "ppo_mc_vf_g1_0_sparse --num-steps 0 --gamma 1.0 --sparse --gae-lambda 1.0"
)

all_commands=()

for seed in {1..5}; do
  for ENV in "${envs[@]}"; do
    for cond in "${CONDITIONS[@]}"; do
      exp_name="$(echo "$cond" | cut -d' ' -f1)"
      extra="$(echo "$cond" | cut -d' ' -f2-)"
      all_commands+=("${ENV} ${exp_name} ${seed} --env-id ${ENV} --seed ${seed} ${extra}")
    done
  done
done

# ── Dispatch ──────────────────────────────────────────────────────────────────

mkdir -p logs
mkdir -p locks
run_commands=()
for idx in "${!all_commands[@]}"; do
  if (( idx % num_instances != cur_instance )); then continue; fi
  entry="${all_commands[$idx]}"
  ENV="$(echo "$entry" | cut -d' ' -f1)"
  exp_name="$(echo "$entry" | cut -d' ' -f2)"
  seed="$(echo "$entry" | cut -d' ' -f3)"
  rest="$(echo "$entry" | cut -d' ' -f4-)"
  lockfile="locks/${ENV}__${exp_name}__${seed}.lock"

  if compgen -G "runs/${ENV}__${exp_name}__${seed}__*/DONE" > /dev/null 2>&1; then
    echo "Skipping ${ENV}__${exp_name}__${seed} (done)"
    continue
  fi
  if [ -f "$lockfile" ]; then
    echo "Skipping ${ENV}__${exp_name}__${seed} (in-progress or failed — rm $lockfile to retry)"
    continue
  fi
  run_commands+=("touch ${lockfile} && .venv/bin/python algorithm.py ${rest} --exp-name ${exp_name} --track --wandb-group fig3 >> logs/${ENV}__${exp_name}__${seed}.log 2>&1")
done

echo "About to run ${#run_commands[@]}/${#all_commands[@]} experiments (fig3 baselines, jobs=${jobs_per_instance})."
if $dry_run; then exit 0; fi

if (( jobs_per_instance <= 1 )); then
  for cmd in "${run_commands[@]}"; do
    $SHELL -c "$cmd"
    sleep 1
  done
else
  for cmd in "${run_commands[@]}"; do
    $SHELL -c "$cmd" &
    while (( $(jobs -rp | wc -l) >= jobs_per_instance )); do
      wait -n 2>/dev/null || true
    done
  done
  wait
  echo "All parallel jobs finished."
fi
