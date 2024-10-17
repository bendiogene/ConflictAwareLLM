#!/bin/bash

# List of config files
configs=(
    "configs/gpt2-xl_experiment2_1_withB.yml"
    "configs/gpt2-xl_experiment2_1_withB_10xNeurons.yml"
    "configs/gpt2-xl_experiment2_1_withB_10xlr.yml"
    "configs/gpt2-xl_experiment2_1_withB_best_lr.yml"
    "configs/gpt2-xl_experiment2_1_withBepochs50s.yml"
)

# Iterate over each config file and run the command
for config in "${configs[@]}"
do
    accelerate launch experiments_scripts/exp_2_1_with_B.py --config "$config"
done