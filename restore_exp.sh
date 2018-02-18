#!/bin/bash

DATA_DIR='path/to/data/'
SAVED_MODELS_DIR='path/to/saved_models/'
LOG_DIR='path/to/training_logs/'

# restore from previously saved shard and train on the next shard

for i in {0..994} # <---- set the loop range to start at previously saved shard
do
    let j=$i+1
    python multi-gpu-train.py --data_dir=$DATA_DIR \
    --restore_path=$SAVED_MODELS_DIR$i \
    --shard=${j} \
    --saved_models_dir=$SAVED_MODELS_DIR \
    --log_dir=$LOG_DIR
done
