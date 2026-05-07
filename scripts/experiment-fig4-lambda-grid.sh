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
reverse_order=false
random_order=false

while [ "$#" -gt 0 ]; do
  case "$1" in
    --jobs)               jobs_per_instance="$2";   shift 2 ;;
    --dry-run)            dry_run=true;               shift ;;
    --reverse)            reverse_order=true;         shift ;;
    --random)             random_order=true;          shift ;;
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
      exp_name="ppo__g1_0__n256__a${atag}__c${ctag}__sparse"
      all_commands+=("${ENV} ${exp_name} ${seed} --env-id ${ENV} --seed ${seed} --num-steps 256 --gamma 1.0 --sparse --gae-lambda-actor ${la} --gae-lambda-critic ${lc}")
    done
  done
done

# ── Dispatch ──────────────────────────────────────────────────────────────────


wandb_runs="$(mktemp)"
echo "Fetching finished/running runs from wandb..."
.venv/bin/python scripts/wandb_fetch_runs.py --list > "$wandb_runs" 2>/dev/null || true

run_commands=()
run_names=()
for idx in "${!all_commands[@]}"; do
  if (( idx % num_instances != cur_instance )); then continue; fi
  entry="${all_commands[$idx]}"
  ENV_="$(echo "$entry" | cut -d' ' -f1)"
  exp_name="$(echo "$entry" | cut -d' ' -f2)"
  seed="$(echo "$entry" | cut -d' ' -f3)"
  rest="$(echo "$entry" | cut -d' ' -f4-)"
  run_dir="runs/${ENV_}__${exp_name}__${seed}"
  run_name="${ENV_}__${exp_name}__${seed}"

  if [ -f "${run_dir}/DONE" ]; then
    echo "Skipping ${run_name} (done)"
    continue
  fi
  if grep -qxF "${run_name}" "$wandb_runs" 2>/dev/null; then
    echo "Skipping ${run_name} (finished in wandb)"
    continue
  fi
  if [ -f "${run_dir}/LOCK" ]; then
    echo "Skipping ${run_name} (in-progress or failed — rm ${run_dir}/LOCK to retry)"
    continue
  fi
  run_commands+=("mkdir -p ${run_dir} && touch ${run_dir}/LOCK && .venv/bin/python algorithm.py ${rest} --exp-name ${exp_name} --track --wandb-group fig4 >> ${run_dir}/run.log 2>&1")
  run_names+=("${run_name}")
done

# Reorder if requested
if $random_order; then
  indices=($(shuf -i 0-$(( ${#run_commands[@]} - 1 )) 2>/dev/null || true))
  if (( ${#indices[@]} > 0 )); then
    _cmds=(); _names=()
    for i in "${indices[@]}"; do _cmds+=("${run_commands[$i]}"); _names+=("${run_names[$i]}"); done
    run_commands=("${_cmds[@]}"); run_names=("${_names[@]}")
  fi
elif $reverse_order; then
  _cmds=(); _names=()
  for (( i=${#run_commands[@]}-1; i>=0; i-- )); do
    _cmds+=("${run_commands[$i]}"); _names+=("${run_names[$i]}")
  done
  run_commands=("${_cmds[@]}"); run_names=("${_names[@]}")
fi

echo "About to run ${#run_commands[@]}/${#all_commands[@]} experiments (fig4 lambda-grid, jobs=${jobs_per_instance})."
if $dry_run; then exit 0; fi

if (( jobs_per_instance <= 1 )); then
  for i in "${!run_commands[@]}"; do
    if .venv/bin/python scripts/wandb_fetch_runs.py --check "${run_names[$i]}" 2>/dev/null; then
      echo "Skipping ${run_names[$i]} (now active in wandb)"
      continue
    fi
    $SHELL -c "${run_commands[$i]}"
    sleep 1
  done
else
  for i in "${!run_commands[@]}"; do
    if .venv/bin/python scripts/wandb_fetch_runs.py --check "${run_names[$i]}" 2>/dev/null; then
      echo "Skipping ${run_names[$i]} (now active in wandb)"
      continue
    fi
    $SHELL -c "${run_commands[$i]}" &
    while (( $(jobs -rp | wc -l) >= jobs_per_instance )); do
      wait -n 2>/dev/null || true
    done
  done
  wait
  echo "All parallel jobs finished."
fi
