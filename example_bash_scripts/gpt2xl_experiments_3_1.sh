#!/bin/bash

# List of config files
configs=(
    "configs/gpt2-xl_experiment3_1_20000_1000.yml"
    "configs/gpt2-xl_experiment3_1_20000_1000_10x_lr.yml"
    "configs/gpt2-xl_experiment3_1_2000_1000.yml"
    "configs/gpt2-xl_experiment3_1_2000_1000_10x_lr.yml"
    "configs/gpt2-xl_experiment3_1_20000_100.yml"
    "configs/gpt2-xl_experiment3_1_20000_1000_10_small_lr.yml"
    "configs/gpt2-xl_experiment3_1_2000_100.yml"
    "configs/gpt2-xl_experiment3_1_2000_1000_10_small_lr.yml"
)

# Iterate over each config file and run the command
for config in "${configs[@]}"
do
    accelerate launch experiments_scripts/exp_3_1.py --config "$config"
done
