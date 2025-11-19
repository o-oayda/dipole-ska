#!/bin/bash
set -euo pipefail

trap 'echo "Command failed at line $LINENO: $BASH_COMMAND" >&2; exit 1' ERR

python_path=$(command -v python3.12 || true)
if [ -z "$python_path" ]; then
    echo "python3.12 not found on PATH; please install or set PYTHON_BIN" >&2
    exit 1
fi

if [ "${1:-}" == "conda" ]; then
    echo "creating conda environment cenv"
    conda_path=$(command -v conda || true)
    if [ -z "$conda_path" ]; then
        echo "Conda not found. Please install before running with conda flag."
    else
        if "$conda_path" env list | awk 'NR>2 {print $1}' \
            | grep -Fxq "cenv"; then
            echo "Conda env 'cenv' exists; remove or choose another name." >&2
            exit 1
        fi
        $conda_path create -n cenv python=3.12 -y
        source "$($conda_path info --base)/etc/profile.d/conda.sh"
        conda activate cenv
        pip install --upgrade pip
        pip install -e .
        conda install -c conda-forge namaster -y
    fi
else
    echo "creating virtual environment .venv"
    if [ -d ".venv" ]; then
        echo ".venv exists; remove or rename it before rerunning." >&2
        exit 1
    fi
    $python_path -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e . 
fi
