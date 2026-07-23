#!/bin/bash

# Per-rank environment shim for mpirun on DCC. Used as:
#
#   mpirun -np N ... scripts/dcc/mpi_rank_env.sh python -u some_script.py --flags
#
# WHY THIS EXISTS
#
# DCC's Slurm exposes only the 'pmi2' MPI plugin (check with 'srun --mpi=list'), and
# the OpenMPI/4.1.6 module was not built against it: it ships only the pmix3x, flux and
# isolated pmix components and links no libpmi. Direct-launching an MPI program with
# srun therefore aborts in MPI_Init_thread with
#
#   OPAL ERROR: Unreachable in file pmix3x_client.c at line 111
#   The application appears to have been direct launched using "srun", but OMPI was
#   not built with SLURM's PMI support and therefore cannot execute.
#
# Verified: 'srun' and 'srun --mpi=pmi2' both fail; 'mpirun' works. So the DCC scripts
# launch with mpirun where the Perlmutter scripts use srun.
#
# The catch: mpirun sets OMPI_COMM_WORLD_{RANK,SIZE,LOCAL_RANK} correctly but leaves
# SLURM_PROCID and SLURM_LOCALID at 0 on EVERY rank. Most of HydraGNN checks the OMPI
# variables first and is unaffected, but three places read the Slurm ones with no OMPI
# fallback:
#
#   hydragnn/utils/datasets/distdataset.py:108,111  int(os.getenv("SLURM_LOCALID","0"))
#   examples/qmugs/qmugs_densmat_inference.py:120   tries SLURM_PROCID first
#   examples/qmugs/qmugs.py:1683                    tries SLURM_PROCID first
#
# Left alone, every rank would believe it is rank 0 and bind to GPU 0 -- silently
# wrong results rather than a crash. Remapping here fixes all three at once and keeps
# the Python identical to Perlmutter's.
export SLURM_PROCID="${OMPI_COMM_WORLD_RANK}"
export SLURM_LOCALID="${OMPI_COMM_WORLD_LOCAL_RANK}"
export SLURM_NPROCS="${OMPI_COMM_WORLD_SIZE}"

exec "$@"
