#!/bin/bash

DATA_DIR='/path/to/data/'
SAVED_MODELS_DIR='path/to/saved_models/'
LOG_DIR='path/to/training_logs/'

for i in {0..993}
do
  if (( $i == 0 ))
  then
    python multi-gpu-train.py --data_dir=$DATA_DIR \
    --shard=${i} \
    --saved_models_dir=$SAVED_MODELS_DIR \
    --log_dir=$LOG_DIR
  else

    let a=$i-1
    python multi-gpu-train.py --data_dir=$DATA_DIR \
    --shard=${i} \
    --saved_models_dir=$SAVED_MODELS_DIR \
    --log_dir=$LOG_DIR \
    --restore_path=$SAVED_MODELS_DIR$a
  fi

done
