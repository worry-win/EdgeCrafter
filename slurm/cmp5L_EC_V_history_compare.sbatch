#!/usr/bin/env bash
#SBATCH --partition=debug,batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
set -euo pipefail
RUN=${1:?locked replay root required}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/cobot/miniforge3/envs/lw-detr/bin/python "$RUN/submission/compare_historical_v_replay.py" "$RUN"
