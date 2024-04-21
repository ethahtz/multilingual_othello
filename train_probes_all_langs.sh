#!/bin/bash

# Usage: ./train_probes_all_langs.sh [config_path] [num_langs]
# config_path: name of the experiment
# num_langs: number of languages to train probes for
# e.g. ./train_probes_all_langs.sh ./exp_configs/naive_training_with_10_atomic_languages.txt 10

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters! Usage: ./train_probes_all_langs.sh [config_path] [num_langs]"
fi

CONFIG_PATH=$1
NUMLANGS=$2

for i in $(seq 0 $NUMLANGS);

do
    python3 train_probe_othello.py --twolayer --exp_config $CONFIG_PATH --token_space_idx $i
done
