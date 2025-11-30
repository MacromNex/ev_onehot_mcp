#!/bin/bash

input_dir=$1
choice=$2
script_dir=$(dirname "$0")

# 5 fold cross eval for ev-onehot
if [ "$choice" == "eval" ];
then
    python $script_dir/train.py $input_dir --cross_val
fi

# Select best seed for ev-onehot

if [ "$choice" == "seed" ];
then
    echo 'Select the best data split, i.e. selecting seed'
    for s in {1..20};
    do
        python $script_dir/train.py $input_dir -s $s
    done
fi


#==== Train the selected model model
if [ "$choice" == "final" ];
then
    seed=$3
    echo "Train the final model with seed $seed"
    python $script_dir/train.py $input_dir -s $seed
fi