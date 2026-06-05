#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=0:60:0
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --constraint=cpu

# need to reformulate the header to fit requirements on specific machines
srun python examples/qmugs/utils/download_data.py \
    --start 0 \
    --end 99 \
    --parallel \
    --max_workers 100 \
    --download_wfns_only