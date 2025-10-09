#!/usr/bin/env bash
set -euo pipefail

JOB_SCRIPT="your_job_script.sh"  # Path to your job script


CONFIG_DIR="your_config_dir"  # Directory containing your YAML config files


LOG_DIR="your_log_dir"  # Directory to store log files
mkdir -p "$LOG_DIR"


YAML_NAMES=(
  D_42
)


for name in "${YAML_NAMES[@]}"; do
  cfg="${CONFIG_DIR}/${name}.yaml"

  if [[ ! -f "$cfg" ]]; then
    echo "Skipping, config not found: $cfg"
    continue
  fi

  log_file="${LOG_DIR}/EXP_${name}_%j.log"
  job_name="EXP_${name}"

  echo "Submitting job ${job_name} with config ${cfg}"
  sbatch \
    --job-name="$job_name" \
    --output="$log_file" \
    "$JOB_SCRIPT" "$cfg"
done
