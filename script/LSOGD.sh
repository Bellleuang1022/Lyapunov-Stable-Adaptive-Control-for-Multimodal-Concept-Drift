#!/usr/bin/env bash

#SBATCH --mail-user=xxx
#SBATCH --mail-type=ALL
#SBATCH --time=1-08:00:00   
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=10            
#SBATCH --mem-per-cpu=11gb            
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4              
#SBATCH --qos=xxx


if [ $# -lt 1 ]; then
  echo "Usage: $0 /path/to/your_config.yaml"
  exit 1
fi
CONFIG_YAML_FILE="$1"

echo "--------------------------------------------------------------------"
echo "  SLURM Job ID   : $SLURM_JOB_ID"
echo "  Job Name       : $SLURM_JOB_NAME"
echo "  Running on     : $(hostname)"
echo "  GPUs assigned  : $CUDA_VISIBLE_DEVICES"
echo "  Config file    : $CONFIG_YAML_FILE"
echo "  Start Time     : $(date)"
echo "--------------------------------------------------------------------"
start_time=$(date +%s)


module purge
module load conda cuda
module list

CONDA_ENV_PATH="your_env_path"
source activate "$CONDA_ENV_PATH"
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment: $CONDA_ENV_PATH"
    exit 1
fi
echo "Activated conda env: $CONDA_ENV_PATH"

export XDG_CACHE_HOME="your_cache_path"
export HF_HOME="your_hf_home_path"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$XDG_CACHE_HOME"
export HF_ACCESS_TOKEN="your_hf_access_token"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_DISTRIBUTED_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

CODE_BASE_DIR="your_code_base_dir"   
MAIN_SCRIPT_PY="main_D.py"

RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_RUN_NAME="EXP_${RUN_TIMESTAMP}"
OUTPUT_PARENT_DIR="your_output_parent_dir"  

echo "--- Using config: $CONFIG_YAML_FILE ---"
cat "$CONFIG_YAML_FILE"
echo "----------------------------------------"

NUM_GPUS_REQUESTED=4  # matches your --gres
COMMAND_PREFIX="python"
if [ "$NUM_GPUS_REQUESTED" -gt 1 ]; then
  echo "Multi‐GPU detected ($NUM_GPUS_REQUESTED); launching with torchrun"
  COMMAND_PREFIX="torchrun --nproc_per_node=$NUM_GPUS_REQUESTED --nnodes=1 --node_rank=0"
else
  echo "Single‐GPU or DDP disabled; using python"
fi

echo "Launch command:"
echo "  $COMMAND_PREFIX $CODE_BASE_DIR/$MAIN_SCRIPT_PY --config $CONFIG_YAML_FILE"

$COMMAND_PREFIX "$CODE_BASE_DIR/$MAIN_SCRIPT_PY" \
    --config "$CONFIG_YAML_FILE" \
    --run-name "$EXPERIMENT_RUN_NAME" \
    --save-path "$OUTPUT_PARENT_DIR"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "Script failed with exit code $EXIT_CODE"
else
  echo "Script finished successfully"
fi

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "\nTotal elapsed: %02d:%02d:%02d (hh:mm:ss)\n" \
  $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
echo "Output parent dir: $OUTPUT_PARENT_DIR/$EXPERIMENT_RUN_NAME"
echo "--------------------------------------------------------------------"

exit $EXIT_CODE
