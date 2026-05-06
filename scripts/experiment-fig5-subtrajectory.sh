#!/usr/bin/env bash
# Fig 5 — Subtrajectory learning: PPO sparse γ=1, N sweep vs GRPO
#
# (1 + 8 N values) × 5 seeds = 45 runs.  Humanoid-v4 sparse, γ=1.
# grpo_g1_0_sparse runs are shared with fig1 and fig3 — skipped automatically.
#
# λ_actor=1 gives MC-like advantages for a fair comparison with GRPO.
# λ_critic=0 gives pure TD(0) bootstrapping, maximising credit-assignment
# benefit at short horizons.
#
# The N sweep shows that small N enables T/N critic updates per episode,
# propagating terminal rewards back through the trajectory and recovering
# or exceeding GRPO sample efficiency.
#
# Usage:
#   Single GPU:  ./scripts/experiment-fig5-subtrajectory.sh 1 0
#   With jobs:   ./scripts/experiment-fig5-subtrajectory.sh 1 0 --jobs 9
#   Dry run:      ./scripts/experiment-fig5-subtrajectory.sh --dry-run

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
n_values=(16 32 64 128 256 512 1024 2048 4096)

all_commands=()

for seed in {1..5}; do
  # GRPO reference — exp_name matches fig1/fig3, skipped automatically if already done.
  all_commands+=("${ENV} grpo_g1_0_sparse ${seed} --env-id ${ENV} --seed ${seed} --grpo --sparse")

  for N in "${n_values[@]}"; do
    # Tag convention: 1.0 -> 1_0, 0.0 -> 0_0 (matches fig4 naming so N=16 run is shared).
    exp_name="ppo_g1_0_n${N}_a1_0_c0_0_sparse"
    all_commands+=("${ENV} ${exp_name} ${seed} --env-id ${ENV} --seed ${seed} --num-steps ${N} --gamma 1.0 --sparse --gae-lambda-actor 1.0 --gae-lambda-critic 0.0")
  done
done

# ── Dispatch ──────────────────────────────────────────────────────────────────

mkdir -p logs
run_commands=()
for idx in "${!all_commands[@]}"; do
  if (( idx % num_instances != cur_instance )); then continue; fi
  entry="${all_commands[$idx]}"
  ENV_="$(echo "$entry" | cut -d' ' -f1)"
  exp_name="$(echo "$entry" | cut -d' ' -f2)"
  seed="$(echo "$entry" | cut -d' ' -f3)"
  rest="$(echo "$entry" | cut -d' ' -f4-)"

  if compgen -G "runs/${ENV_}__${exp_name}__${seed}__*/DONE" > /dev/null 2>&1; then
    echo "Skipping ${ENV_}__${exp_name}__${seed} (already done)"
    continue
  fi
  run_commands+=(".venv/bin/python algorithm.py ${rest} --exp-name ${exp_name} --track --wandb-group fig5 >> logs/${ENV}__${exp_name}__${seed}.log 2>&1")
done

echo "About to run ${#run_commands[@]}/${#all_commands[@]} experiments (fig5 subtrajectory, jobs=${jobs_per_instance})."
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
