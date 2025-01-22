#!/bin/bash

# List of config files
configs=(
    "configs/gpt2-xl_experiment2_1.yml"
    "configs/gpt2-xl_experiment2_1_10xNeurons.yml"
    "configs/gpt2-xl_experiment2_1_10xlr.yml"
    "configs/gpt2-xl_experiment2_1_best_lr.yml"
    "configs/gpt2-xl_experiment2_1epochs50s.yml"
)

# Iterate over each config file and run the command
for config in "${configs[@]}"
do
    accelerate launch experiments_scripts/exp_2_1.py --config "$config"
done