#!/bin/bash
# ============================================================
# GAN Distribution Recovery Experiment - Environment Setup
# ============================================================
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
#
# Prerequisites: conda (Anaconda or Miniconda) installed
# ============================================================

set -e

ENV_NAME="gan_experiment"

echo "============================================"
echo "Setting up conda environment: $ENV_NAME"
echo "============================================"

# Remove existing environment if present
conda env remove -n $ENV_NAME -y 2>/dev/null || true

# Create environment from YAML
conda env create -f environment.yml

echo ""
echo "============================================"
echo "Environment created successfully!"
echo ""
echo "To activate:   conda activate $ENV_NAME"
echo "To run:        python gan_experiment.py"
echo "============================================"
