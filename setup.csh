#!/bin/csh

set python_path=`which python3.12`

if ( "$1" == "conda" ) then
    echo "creating both conda environment cenv and venv"
    set conda_path=`which conda`
    if ( "$conda_path" == "" ) then
        echo "Conda not found. Please install before running with conda flag."
    else
        $conda_path create -n cenv python=3.12 -y
        source `$conda_path info --base`/etc/profile.d/conda.csh
        conda activate cenv
        pip install --upgrade pip
        pip install -e .

        echo "creating virtual environment .venv"
        $python_path -m venv .venv
        source .venv/bin/activate.csh
        pip install --upgrade pip
        pip install -e .
    endif
else
    echo "creating virtual environment .venv"
    $python_path -m venv .venv
    source .venv/bin/activate.csh
    pip install --upgrade pip
    pip install -e . 
endif