#!/usr/bin/env bash
# Fig 1 — Setup validation: PPO dense vs PPO sparse vs GRPO
#
# 3 conditions × 3 envs × 5 seeds = 45 runs.
#
# Conditions (exp-name encodes condition; run dirs are {env}__{exp_name}__{seed}__{ts}):
#   grpo_g1_0_sparse       GRPO: episodic MC, no VF, group_mean+std, γ=1, sparse
#   ppo_g0_99_n2048_dense  PPO default: γ=0.99, λ=0.95, N=2048, dense
#   ppo_g0_99_n2048_sparse PPO sparse:  γ=0.99, λ=0.95, N=2048, sparse
#
# Seeds loop outermost so partial results (any completed seed) are plottable.
#
# Usage:
#   Single GPU:  ./scripts/experiment-fig1-dense-vs-sparse.sh 1 0
#   Two GPUs:    CUDA_VISIBLE_DEVICES=1 ./scripts/experiment-fig1-dense-vs-sparse.sh 2 1 &
#   With jobs:   ./scripts/experiment-fig1-dense-vs-sparse.sh 1 0 --jobs 9
#   Dry run:      ./scripts/experiment-fig1-dense-vs-sparse.sh --dry-run

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

# Each entry: "<exp_name> <extra args>"
# Tag convention: float values use dot-to-underscore (0.99 -> 0_99, 1.0 -> 1_0)
# so exp_names match across figures and skip logic deduplicates correctly.
declare -a CONDITIONS=(
  "grpo_g1_0_sparse --grpo --sparse"
  "ppo_g0_999_n2048_dense"
  "ppo_g0_999_n2048_sparse --sparse"
  "ppo_g0_999_n256_dense --num-steps 256"
  "ppo_g0_999_n256_sparse --num-steps 256 --sparse"
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

mkdir -p logs locks
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
  run_commands+=("touch ${lockfile} && .venv/bin/python algorithm.py ${rest} --exp-name ${exp_name} --track --wandb-group fig1 >> logs/${ENV}__${exp_name}__${seed}.log 2>&1")
done

echo "About to run ${#run_commands[@]}/${#all_commands[@]} experiments (fig1, jobs=${jobs_per_instance})."
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
