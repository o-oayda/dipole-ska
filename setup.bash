#!/bin/bash
set -euo pipefail

trap 'echo "Command failed at line $LINENO: $BASH_COMMAND"; exit 1' ERR

python_path=$(which python3.12)

if [ "${1:-}" == "conda" ]; then
    echo "creating conda environment cenv"
    conda_path=$(which conda || true)
    if [ -z "$conda_path" ]; then
        echo "Conda not found. Please install before running with conda flag."
    else
        $conda_path create -n cenv python=3.12 -y
        source "$($conda_path info --base)/etc/profile.d/conda.sh"
        conda activate cenv
        pip install --upgrade pip
        pip install -e .
        conda install -c conda-forge namaster -y
    fi
else
    echo "creating virtual environment .venv"
    $python_path -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e . 
fi
