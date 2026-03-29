#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") -c <config> [-n <num_shards>] [-o <output>] [-- <extra vla-eval run args>]

Run a benchmark in parallel shards and merge results.

Options:
  -c <config>       Config YAML file (required)
  -n <num_shards>   Number of shards (default: 50)
  -o <output>       Output file for merged results (default: results/<config_name>.json)
  -h                Show this help

Extra arguments after -- are passed through to each 'vla-eval run' invocation.
Example:
  $(basename "$0") -c configs/libero_10.yaml -n 4 -- --gpus 4 --save-traj --server-url ws://0.0.0.0:8001
EOF
  exit "${1:-0}"
}

CONFIG=""
NUM_SHARDS=50
OUTPUT=""
EXTRA_ARGS=()

# Split args at -- : before goes to getopts, after goes to vla-eval run
pre_args=()
found_sep=false
for arg in "$@"; do
  if [[ "$arg" == "--" ]]; then
    found_sep=true
    continue
  fi
  if $found_sep; then
    EXTRA_ARGS+=("$arg")
  else
    pre_args+=("$arg")
  fi
done

# Parse our own flags from pre_args
set -- "${pre_args[@]+"${pre_args[@]}"}"
while getopts "c:n:o:h" opt; do
  case "$opt" in
    c) CONFIG="$OPTARG" ;;
    n) NUM_SHARDS="$OPTARG" ;;
    o) OUTPUT="$OPTARG" ;;
    h) usage 0 ;;
    *) usage 1 ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Error: -c <config> is required." >&2
  usage 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Error: config file not found: $CONFIG" >&2
  exit 1
fi

# Derive output name from config filename if not specified
if [[ -z "$OUTPUT" ]]; then
  config_name="$(basename "$CONFIG" .yaml)"
  config_name="$(basename "$config_name" .yml)"
  OUTPUT="results/${config_name}.json"
fi

cleanup() {
  echo "Cleaning up background processes..."
  kill -- -$$ 2>/dev/null || true
}
trap cleanup EXIT

echo "Config:     $CONFIG"
echo "Shards:     $NUM_SHARDS"
echo "Output:     $OUTPUT"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "Extra args: ${EXTRA_ARGS[*]}"
fi
echo ""

# Check for existing shard results
existing=$(CONFIG="$CONFIG" NUM_SHARDS="$NUM_SHARDS" python3 -c "
import os, yaml, re
from pathlib import Path
with open(os.environ['CONFIG']) as f:
    cfg = yaml.safe_load(f)
num_shards = os.environ['NUM_SHARDS']
output_dir = Path(cfg.get('output_dir', './results'))
found = []
seen = set()
for b in cfg.get('benchmarks', []):
    name = b.get('name') or b['benchmark'].rsplit(':', 1)[-1]
    sub = b.get('subname')
    if sub:
        name = f'{name}_{sub}'
    safe = re.sub(r'[^\w\-.]', '_', name)
    if safe in seen:
        continue
    seen.add(safe)
    found.extend(output_dir.glob(f'{safe}_shard*of{num_shards}.json'))
if found:
    print(f'{len(found)} existing shard file(s) found, e.g.: {found[0]}')
")
if [[ -n "$existing" ]]; then
  echo "Error: $existing" >&2
  echo "Remove existing results or use a different output_dir." >&2
  exit 1
fi

# If --save-traj is in extra args but --traj-name is not, generate a shared
# traj-name so all shards write to the same trajectory directory.
has_save_traj=false
has_traj_name=false
traj_name=""
next_is_traj_name=false
for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
  if $next_is_traj_name; then
    traj_name="$arg"
    next_is_traj_name=false
    continue
  fi
  case "$arg" in
    --save-traj) has_save_traj=true ;;
    --traj-name) has_traj_name=true; next_is_traj_name=true ;;
  esac
done
if $has_save_traj && ! $has_traj_name; then
  traj_name="$(date -u +%Y%m%d_%H%M%S)"
  EXTRA_ARGS+=("--traj-name" "$traj_name")
fi

echo "Launching ${NUM_SHARDS} shards..."

pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  vla-eval run -c "$CONFIG" --shard-id "$i" --num-shards "$NUM_SHARDS" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
  pids+=($!)
done

echo "Waiting for all shards to finish..."
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=$((failed + 1))
  fi
done

if [[ "$failed" -gt 0 ]]; then
  echo "ERROR: $failed of $NUM_SHARDS shards failed." >&2
  exit 1
fi

echo "Merging results..."
if $has_save_traj; then
  vla-eval merge -c "$CONFIG" --traj-name "$traj_name" -o "$OUTPUT"
  echo "Merging trajectory shards..."
  vla-eval merge-traj -c "$CONFIG" --traj-name "$traj_name"
else
  vla-eval merge -c "$CONFIG" -o "$OUTPUT"
fi

echo "Done. Results saved to $OUTPUT"
