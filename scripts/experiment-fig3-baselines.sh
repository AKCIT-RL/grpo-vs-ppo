#!/usr/bin/env bash
# Fig 3 — Role of the value function as a baseline
#
# 3 conditions × 3 envs × 5 seeds = 45 runs.
# grpo_g1_0_sparse runs are shared with fig1 and fig5 — skipped automatically.
#
# Conditions:
#   grpo_sparse               GRPO: no VF, group_mean+std, γ=1, sparse (shared with fig1, fig5)
#   ppo_g1_0_n0_a1_0_c1_0_sparse  GAE λ=1 (episodic) ≡ MC returns with VF as baseline, γ=1, sparse
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

envs=(Humanoid-v4 Hopper-v4 Walker2d-v4)

declare -a CONDITIONS=(
  "grpo_sparse --grpo --sparse"
  "ppo__g1_0__n0__a1_0__c1_0__sparse --num-steps 0 --gamma 1.0 --sparse --gae-lambda 1.0"
)

all_commands=()

for seed in {1..5}; do
  for ENV in "${envs[@]}"; do
    for cond in "${CONDITIONS[@]}"; do
      exp_name="${cond%% *}"
      extra="${cond#* }"
      if [ "$extra" = "$cond" ]; then extra=""; fi
      all_commands+=("${ENV} ${exp_name} ${seed} --env-id ${ENV} --seed ${seed} ${extra}")
    done
  done
done

# ── Dispatch ──────────────────────────────────────────────────────────────────


wandb_runs="$(mktemp)"
echo "Fetching finished runs from wandb..."
.venv/bin/python scripts/wandb_fetch_runs.py --sync > "$wandb_runs" 2>/dev/null || true

run_commands=()
run_names=()
for idx in "${!all_commands[@]}"; do
  if (( idx % num_instances != cur_instance )); then continue; fi
  entry="${all_commands[$idx]}"
  ENV="$(echo "$entry" | cut -d' ' -f1)"
  exp_name="$(echo "$entry" | cut -d' ' -f2)"
  seed="$(echo "$entry" | cut -d' ' -f3)"
  rest="$(echo "$entry" | cut -d' ' -f4-)"
  run_dir="runs/${ENV}__${exp_name}__${seed}"
  run_name="${ENV}__${exp_name}__${seed}"

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
  run_commands+=("mkdir -p ${run_dir} && touch ${run_dir}/LOCK && .venv/bin/python algorithm.py ${rest} --exp-name ${exp_name} --track --wandb-group fig3 >> ${run_dir}/run.log 2>&1")
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

echo "About to run ${#run_commands[@]}/${#all_commands[@]} experiments (fig3 baselines, jobs=${jobs_per_instance})."
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
