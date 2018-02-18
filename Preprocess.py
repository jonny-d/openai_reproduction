import tensorflow as tf
import numpy as np
import os
import io

def read_and_split_data(data_dir, batch_size, seq_length, num_gpus):
    """split data equally between each GPU tower for 'persistent' data parrallelism

    Keyword arguments:
    data_dir -- Path to directory containing the input.txt file
    batch_size -- hpyerparameter
    seq_length -- hyperparameter. This is divided by num_gpus
    num_gpus -- number of GPU 'towers'
    """
    print('Preprocessing data...')

    input_file = os.path.join(data_dir, "input.txt")

    # read the input file as bytes
    with io.open(input_file, 'rb') as f:
        shard = np.array(bytearray(f.read()), dtype='int32')

    batch_size_per_tower = int(batch_size / num_gpus)

    num_updates = int(shard.size / (seq_length * batch_size))

    shard = shard[:num_updates * seq_length * batch_size]

    # shift the data one step to get the labels for the language modelling task
    xdata = shard
    ydata = np.copy(shard)
    ydata[:-1] = xdata[1:]
    ydata[-1] = xdata[0]

    # reshape the data so that each row is a subsequence (batch_size, length_of_each_sub_sequence)
    x_batches = xdata.reshape((batch_size, num_updates * seq_length))
    y_batches = ydata.reshape((batch_size, num_updates * seq_length))

    # split the data so that there is an equal number of batches for each tower
    x_towers = np.split(x_batches, num_gpus, axis=0)
    y_towers = np.split(y_batches, num_gpus, axis=0)

    dataset_list =[]

    for i in range(num_gpus):

        # process each tower's data into batches with shape (num_updates, batch_size_per_tower, seq_length_per_tower)
        x_tower = np.concatenate(np.split(x_towers[i], num_updates, axis=1),axis=0)
        y_tower = np.concatenate(np.split(y_towers[i], num_updates, axis=1),axis=0)

        x_tower = x_tower.reshape(num_updates, batch_size_per_tower, seq_length)
        y_tower = y_tower.reshape(num_updates, batch_size_per_tower, seq_length)

        # using the TF Dataset API
        dataset = tf.data.Dataset.from_tensor_slices((x_tower, y_tower))
        dataset_list.append(dataset)

    print('Done')

    return dataset_list, num_updates
