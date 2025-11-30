#!bin/bash
# Usage: bash run_script.sh <data_dir> <protein>
# Example: bash run_script.sh examples/Savinase/activity Savinase

data_dir=$1
protein=$2
# prepare wild-type sequence in $data_dir
## wt.fasta

build_sh_dir=$(dirname "$0")
# Build EV+Onehot model
## Activate environment
# conda activate prot-fit-env

# ## Create alignment using jackhammer.sh
bash $build_sh_dir/scripts/jackhmmer.sh $data_dir  0.5 10  data/uniref100/uniref100.fasta

# Move files to align folder
mkdir -p $data_dir/align
mv $data_dir/alignment* $data_dir/align
mv $data_dir/target* $data_dir/align
mv $data_dir/iter-* $data_dir/align

# ## Create EVmutation model
bash $build_sh_dir/scripts/plmc.sh $protein $data_dir

# Prepare dataset in $data_dir
# data.csv, columns: seq,log_fitness

# Train EV+onehot model
# python $build_sh_dir/train.py $data_dir

# Evaluate the best data split
# bash $build_sh_dir/run_script.sh examples/Savinase_casein Savinase