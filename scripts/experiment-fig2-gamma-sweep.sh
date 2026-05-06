#!/usr/bin/env bash
# Fig 2 — Gamma sweep: PPO sparse, γ ∈ {0.9, 0.95, 0.99, 1.0} across 3 envs
#
# 4 gammas × 3 envs × 5 seeds = 60 runs.
# ppo_g0_99_n2048_sparse (gamma=0.99) runs are shared with fig1 — skipped automatically.
#
# Shows vanishing advantage propagation for γ<1 in long-horizon sparse settings.
#
# Usage:
#   Single GPU:  ./scripts/experiment-fig2-gamma-sweep.sh 1 0
#   With jobs:   ./scripts/experiment-fig2-gamma-sweep.sh 1 0 --jobs 12
#   Dry run:      ./scripts/experiment-fig2-gamma-sweep.sh --dry-run

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

gammas=(0.9 0.95 0.99 0.999 1.0)
envs=(Humanoid-v4 Hopper-v4 Walker2d-v4)

all_commands=()

for seed in {1..5}; do
  for ENV in "${envs[@]}"; do
    for gamma in "${gammas[@]}"; do
      gtag="${gamma//./_}"
      exp_name="ppo_g${gtag}_n2048_sparse"
      all_commands+=("${ENV} ${exp_name} ${seed} --env-id ${ENV} --seed ${seed} --gamma ${gamma} --sparse")
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
  run_commands+=("touch ${lockfile} && .venv/bin/python algorithm.py ${rest} --exp-name ${exp_name} --track --wandb-group fig2 >> logs/${ENV}__${exp_name}__${seed}.log 2>&1")
done

echo "About to run ${#run_commands[@]}/${#all_commands[@]} experiments (fig2 gamma-sweep, jobs=${jobs_per_instance})."
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
