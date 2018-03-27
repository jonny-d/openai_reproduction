DATA_DIR='/path/to/data'
SAVED_MODELS_DIR='path/to/saved_models'
LOG_DIR='path/to/training_logs'



for size in 256 512 1024 2048 4096;
do
  # train on first 5 shards
  for i in $(seq 0 4)
  do
    # initialize weights for first shard
    if (( $i == 0 ))
    then
      (time python multi-gpu-train.py --data_dir=$DATA_DIR/ \
      --shard=${i} \
      --saved_models_dir=$SAVED_MODELS_DIR/$size/ \
      --log_dir=$LOG_DIR/$size/ \
      --rnn_size=$size \
      --timer=$size-time) 2>>$size-time
    else
      # initialise from previous checkpoint
      let a=$i-1
      (time python multi-gpu-train.py --data_dir=$DATA_DIR/ \
      --shard=${i} \
      --saved_models_dir=$SAVED_MODELS_DIR/$size/ \
      --log_dir=$LOG_DIR/$size/ \
      --restore_path=$SAVED_MODELS_DIR/$size/$a \
      --rnn_size=$size \
      --timer=$size-time) 2>>$size-time
    fi

  done
done



