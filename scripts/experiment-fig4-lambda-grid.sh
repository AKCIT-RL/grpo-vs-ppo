#!/usr/bin/env bash
# Fig 4 — Decoupled lambda grid: λ_actor × λ_critic
#
# 4 × 4 grid × 5 seeds = 80 runs.  Humanoid-v4 sparse, γ=1, N=16.
#
# λ_actor ∈ {0.0, 0.5, 0.95, 1.0}  — bias/variance of the actor advantage
# λ_critic ∈ {0.0, 0.5, 0.95, 1.0} — bootstrapping depth for critic targets
#
# Standard GAE couples both under a single λ.  This grid shows that Bellman
# consistency (critic) and MC-like advantages (actor) call for different values.
#
# Usage:
#   Single GPU:  ./scripts/experiment-fig4-lambda-grid.sh 1 0
#   With jobs:   ./scripts/experiment-fig4-lambda-grid.sh 1 0 --jobs 16
#   Dry run:      ./scripts/experiment-fig4-lambda-grid.sh --dry-run

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

ENV="Humanoid-v4"
lambda_actor_values=(0.0 0.5 0.95 1.0)
lambda_critic_values=(0.0 0.5 0.95 1.0)

all_commands=()

for seed in {1..5}; do
  for la in "${lambda_actor_values[@]}"; do
    for lc in "${lambda_critic_values[@]}"; do
      atag="${la//./_}"
      ctag="${lc//./_}"
      # Tag convention matches fig5: 1.0 -> 1_0, 0.0 -> 0_0
      # ppo_g1_0_n16_a1_0_c0_0_sparse is shared with fig5 N=32 entry -- skipped automatically.
      exp_name="ppo_g1_0_n16_a${atag}_c${ctag}_sparse"
      all_commands+=("${ENV} ${exp_name} ${seed} --env-id ${ENV} --seed ${seed} --num-steps 32 --gamma 1.0 --sparse --gae-lambda-actor ${la} --gae-lambda-critic ${lc}")
    done
  done
done

# ── Dispatch ──────────────────────────────────────────────────────────────────

mkdir -p logs locks
run_commands=()
for idx in "${!all_commands[@]}"; do
  if (( idx % num_instances != cur_instance )); then continue; fi
  entry="${all_commands[$idx]}"
  ENV_="$(echo "$entry" | cut -d' ' -f1)"
  exp_name="$(echo "$entry" | cut -d' ' -f2)"
  seed="$(echo "$entry" | cut -d' ' -f3)"
  rest="$(echo "$entry" | cut -d' ' -f4-)"
  lockfile="locks/${ENV_}__${exp_name}__${seed}.lock"

  if compgen -G "runs/${ENV_}__${exp_name}__${seed}__*/DONE" > /dev/null 2>&1; then
    echo "Skipping ${ENV_}__${exp_name}__${seed} (done)"
    continue
  fi
  if [ -f "$lockfile" ]; then
    echo "Skipping ${ENV_}__${exp_name}__${seed} (in-progress or failed — rm $lockfile to retry)"
    continue
  fi
  run_commands+=("touch ${lockfile} && .venv/bin/python algorithm.py ${rest} --exp-name ${exp_name} --track --wandb-group fig4 >> logs/${ENV_}__${exp_name}__${seed}.log 2>&1")
done

echo "About to run ${#run_commands[@]}/${#all_commands[@]} experiments (fig4 lambda-grid, jobs=${jobs_per_instance})."
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
