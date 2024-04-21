#!/bin/bash

# Usage: Usage: ./train_probes_ft_ckpts.sh [config_path]
# config_path: path to the config file
# e.g. ./train_probes_ft_ckpts.sh exp_configs/multiple_lang_types_unified_output.txt

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters! Usage: ./train_probes_ft_ckpts.sh [config_path]"
fi

CONFIG_PATH=$1


filename="${CONFIG_PATH##*/}"  # This strips everything before the last '/' character
exp_name="${filename%.*}" 

# train the probes of the pretrained model
python3 train_probe_othello.py --twolayer --exp_config $CONFIG_PATH

# train the probes of the pretrained model throught the finetuning process
for i in $(ls ./ckpts/gpt_${exp_name}_ft*);
do
    python3 train_probe_othello.py --twolayer --exp_config $CONFIG_PATH --ckpt_dir $i
done
