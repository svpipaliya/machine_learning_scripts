#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=125G
#SBATCH --mail-user=yassine.damergi@epfl.ch
#SBATCH --job-name=XGB_model_selection
#SBATCH --output=XGB_model_selection_out


module load gcc/8.4.0 python/3.7.7
virtualenv --system-site-packages /home/ydamergi/venvs/venv-for-python
source /home/ydamergi/venvs/venv-for-python/bin/activate

pip install --no-cache-dir xlrd==1.2.0
pip install --no-cache-dir numpy
pip install --no-cache-dir matplotlib
pip install --no-cache-dir itertools
pip install --no-cache-dir statistics 
pip install --no-cache-dir sklearn
pip install --no-cache-dir import os
pip install --no-cache-dir import sys
pip install --no-cache-dir import inspect
pip install --no-cache-dir warnings
pip install --no-cache-dir xgboost
pip install --no-cache-dir scikit_optimize
python XGB_model_selection.py