#!/bin/bash

# List of config files
configs=(
    "configs/sparsity_lr/gpt2-xl_experiment2_1_withB_2000_6000_2e-5.yml"
    "configs/sparsity_lr/gpt2-xl_experiment2_1_withB_2000_6000_4e-5.yml"
    "configs/sparsity_lr/gpt2-xl_experiment2_1_withB_2000_6000_8e-5.yml"
	"configs/sparsity_lr/gpt2-xl_experiment2_1_withB_2000_6000_12e-5.yml"
    "configs/sparsity_lr/gpt2-xl_experiment2_1_withB_2000_6000_16e-5.yml"
    "configs/sparsity_lr/gpt2-xl_experiment2_1_2000_6000_3e-4.yml"
    "configs/sparsity_lr/gpt2-xl_experiment2_1_2000_6000_5e-4.yml"
    "configs/sparsity_lr/gpt2-xl_experiment2_1_2000_6000_7e-4.yml"
    "configs/sparsity_lr/gpt2-xl_experiment2_1_2000_6000_1e-3.yml" 
)

# Iterate over each config file and run the command
for config in "${configs[@]}"
do
    accelerate launch experiments_scripts/exp_2_1_with_B.py --config "$config"
done
