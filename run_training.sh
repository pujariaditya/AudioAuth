#!/bin/bash
# Training wrapper script to handle cuDNN library conflicts
# This script ensures PyTorch uses its bundled cuDNN instead of system cuDNN

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Find the virtual environment's site-packages directory
VENV_SITE_PACKAGES="${SCRIPT_DIR}/.venv/lib/python3.10/site-packages"

# IMPORTANT: Override LD_LIBRARY_PATH completely to avoid system cuDNN
# Only use PyTorch's bundled cuDNN 8.7.0 (no system CUDA libraries)
export LD_LIBRARY_PATH="${VENV_SITE_PACKAGES}/nvidia/cudnn/lib"

# Enable lazy CUDA module loading for better compatibility
export CUDA_MODULE_LOADING=LAZY

# Disable cuDNN benchmarking initially (can be enabled in config if needed)
export CUDNN_BENCHMARK=0

# Control joblib/loky to prevent multiprocessing issues
export JOBLIB_BACKEND=threading
export LOKY_MAX_CPU_COUNT=1

# Use thread-based method for wandb to avoid multiprocessing
export WANDB_START_METHOD=thread

# Print environment info for debugging
echo "========================================="
echo "Training Environment Configuration"
echo "========================================="
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "CUDA_MODULE_LOADING: ${CUDA_MODULE_LOADING}"
echo "========================================="

# Change to the script directory to ensure we use the correct environment
cd "${SCRIPT_DIR}"

# Run the training command with uv
# Use --no-project to prevent uv from discovering parent projects
# Pass all arguments to this script to the training command
uv run --no-project torchrun "$@"