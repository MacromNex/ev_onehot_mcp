#!/bin/bash
#
## Estimates couplings model from alignment with plmc package
#
## SBATCH --cluster=<clustername>
## SBATCH --partition=<partitionname>
## SBATCH --account=<accountname>
## SBATCH --job-name=plmc
## SBATCH --output=logs/plmc.out
## SBATCH --gres=gpu:0              # Number of GPU(s) per node.
## SBATCH --cpus-per-task=2         # CPU cores/threads
## SBATCH --mem=4000M              # memory per node
## SBATCH --time=0-24:00            # Max time (DD-HH:MM)
## SBATCH --ntasks=1                # Only set to >1 if you want to use multi-threading
## export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

PROTEIN=$1
DATASET=$2

PLMC_DIR=/home/xux/Desktop/done_projects/Prot_pred/ref_works/0.fitness  # PLMC HOME
    
mkdir -p $DATASET/plmc

$PLMC_DIR/plmc/bin/plmc \
    -o $DATASET/plmc/uniref100.model_params \
    -c $DATASET/plmc/uniref100.EC \
    -f $PROTEIN \
    -le 16.2 -lh 0.01 -m 200 -t 0.2 \
    -g $DATASET/align/alignment.a2m

# ../plmc/bin/plmc -o Subtilisin_BPN/uniref100.model_params -c inference/Subtilisin_BPN/uniref100.EC -f Subtilisin -le 16.2 -lh 0.01 -m 200 -t 0.2  -g alignments/Subtilisin_BPN.a2m
