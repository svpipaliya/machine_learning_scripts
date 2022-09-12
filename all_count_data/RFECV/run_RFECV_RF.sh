#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=12G
#SBATCH	--cpus-per-task=8
#SBATCH --mail-user=yassine.damergi@epfl.ch
#SBATCH --job-name=RFECV_RF
#SBATCH --output=RFECV_RF_out


module load gcc/8.4.0 python/3.7.7
virtualenv --system-site-packages /home/ydamergi/venvs/venv-for-python
source /home/ydamergi/venvs/venv-for-python/bin/activate
pip install --no-cache-dir xlrd==1.2.0
pip install --no-cache-dir numpy
pip install --no-cache-dir sklearn
pip install --no-cache-dir xgboost
pip install --no-cache-dir matplotlib
pip install --no-cache-dir warnings
pip install --no-cache-dir itertools
pip install --no-cache-dir statistics 
pip install --no-cache-dir mlxtend
pip install --no-cache-dir seaborn
pip install --no-cache-dir yellowbrick
python RFECV_RF.py