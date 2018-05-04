# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import argparse
import time
from Preprocess import read_and_split_data
import re

parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default=None,
                    help='Path to training data directory') # not optional
parser.add_argument('--saved_models_dir', type=str, default='saved_models',
                    help='Name of directory to save models during training')
parser.add_argument('--log_dir', type=str, default='training_logs',
                    help='Name of directory for storing losses during training')
parser.add_argument('--rnn_size', type=int, default=20,
                    help='Size of RNN hidden states')
parser.add_argument('--batch_size', type=int, default=128,
                    help='RNN minibatch size')
parser.add_argument('--seq_length', type=int, default=256,
                    help='RNN sequence length')
parser.add_argument('--init_lr', type=float, default=5*10**-4, # value from paper
                    help='Initial learning rate')
parser.add_argument('--embedding_size', type=int, default=64,
                    help='Character embedding layer size')
parser.add_argument('--wn', type=int, default=1,
                    help='Switch for weight normalisation on the mLSTM parameters. Integer argument of 1 for ON and 0 for OFF')
parser.add_argument('--restore_path', type= str, default=None,
                    help='Path to a directory from which to restore a model from previous session')
parser.add_argument('--lr_decay', type=int, default=1,
                    help='Switch for learning rate decay. Integer argument of 1 for ON and 0 for OFF')
parser.add_argument('--lr_decay_steps', type=int, default=None,
                    help='Decay the learning_rate to zero over N steps')
parser.add_argument('--shard', type=str, default='0',
                    help='for Amazon data experiment')
parser.add_argument('--num_gpus', type=int, default=4,
                    help='How many GPUs to use.')
parser.add_argument('--vocab_size', type=int, default=256,
                    help='Byte level model uses 256 dimensional inputs.')

args = parser.parse_args()

rnn_size = args.rnn_size
batch_size = args.batch_size
seq_length = args.seq_length
embedding_size = args.embedding_size
num_gpus = args.num_gpus
vocabulary_size = args.vocab_size # because the inputs are bytes

# Total number of training bytes in the large Amazon dataset ~38.8 Billion
training_bytes = 38800000000

# This is the number of updates over the entire dataset
args.lr_decay_steps = int(training_bytes/(seq_length*batch_size))

data_dir = args.data_dir
data_dir_shard = os.path.join(data_dir,args.shard)

# preprocess the data
dataset_list, num_steps = read_and_split_data(data_dir_shard, batch_size, seq_length, num_gpus)

# this is the length of the sequence seen by each tower since the seq_length is divided by the number of towers
tower_batch_size = int(batch_size / num_gpus)

global nloaded
nloaded = 0

def load_params(shape, dtype, *args, **kwargs):
    'Initialize the weights with values from checkpoint'
    global nloaded
    nloaded += 1
    return weights_list[nloaded - 1]

if args.restore_path is not None:

    # restore the weights from a previous shard and initialize the variables
    weights_list = np.load(args.restore_path + '/model.npy')
    initializer = load_params
else:
    # use the 'Xavier' initialization technique http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    initializer = tf.glorot_normal_initializer()

def inference(inputs):

    # byte embedding
    W_embedding = tf.get_variable('W_embedding', shape=(vocabulary_size, embedding_size), initializer=initializer)

    # mt = (Wmxxt) ⊙ (Wmhht−1) - equation 18
    Wmx = tf.get_variable('Wmx', shape=(embedding_size, rnn_size), initializer=initializer)
    Wmh = tf.get_variable('Wmh', shape=(rnn_size, rnn_size), initializer=initializer )

    # hˆt = Whxxt + Whmmt
    Whx = tf.get_variable('Whx', shape=(embedding_size, rnn_size), initializer=initializer)
    Whm = tf.get_variable('Whm', shape=(rnn_size,rnn_size), initializer=initializer)
    Whb = tf.get_variable('Whb', shape=(1, rnn_size), initializer=initializer)

    # it = σ(Wixxt + Wimmt)
    Wix = tf.get_variable('Wix', shape=(embedding_size, rnn_size), initializer=initializer)
    Wim = tf.get_variable('Wim', shape=(rnn_size, rnn_size), initializer=initializer)
    Wib = tf.get_variable('Wib', shape=(1, rnn_size), initializer=initializer)

    # ot = σ(Woxxt + Wommt)
    Wox = tf.get_variable('Wox', shape=(embedding_size, rnn_size), initializer=initializer)
    Wom = tf.get_variable('Wom', shape=(rnn_size, rnn_size), initializer=initializer)
    Wob = tf.get_variable('Wob', shape=(1, rnn_size), initializer=initializer)

    # ft =σ(Wfxxt +Wfmmt)
    Wfx = tf.get_variable('Wfx', shape=(embedding_size, rnn_size),initializer=initializer)
    Wfm = tf.get_variable('Wfm', shape=(rnn_size, rnn_size), initializer=initializer)
    Wfb = tf.get_variable('Wfb', shape=(1, rnn_size), initializer=initializer)

    # define the g parameters for weight normalization if wn switch is on
    if args.wn == 1:

        gmx = tf.get_variable('gmx', shape=(rnn_size), initializer=initializer)
        gmh = tf.get_variable('gmh', shape=(rnn_size), initializer=initializer)

        ghx = tf.get_variable('ghx', shape=(rnn_size), initializer=initializer)
        ghm = tf.get_variable('ghm', shape=(rnn_size), initializer=initializer)

        gix = tf.get_variable('gix', shape=(rnn_size), initializer=initializer)
        gim = tf.get_variable('gim', shape=(rnn_size), initializer=initializer)

        gox = tf.get_variable('gox', shape=(rnn_size), initializer=initializer)
        gom = tf.get_variable('gom', shape=(rnn_size), initializer=initializer)

        gfx = tf.get_variable('gfx', shape=(rnn_size), initializer=initializer)
        gfm = tf.get_variable('gfm', shape=(rnn_size), initializer=initializer)

        # normalized weights
        Wmx = tf.nn.l2_normalize(Wmx, dim=0)*gmx
        Wmh = tf.nn.l2_normalize(Wmh, dim=0)*gmh

        Whx = tf.nn.l2_normalize(Whx,dim=0)*ghx
        Whm = tf.nn.l2_normalize(Whm,dim=0)*ghm

        Wix = tf.nn.l2_normalize(Wix,dim=0)*gix
        Wim = tf.nn.l2_normalize(Wim,dim=0)*gim

        Wox = tf.nn.l2_normalize(Wox,dim=0)*gox
        Wom = tf.nn.l2_normalize(Wom,dim=0)*gom

        Wfx = tf.nn.l2_normalize(Wfx,dim=0)*gfx
        Wfm = tf.nn.l2_normalize(Wfm,dim=0)*gfm

    # get_variables for saving state across unrolled network.
    saved_output = tf.get_variable('saved_output', initializer=tf.zeros([tower_batch_size, rnn_size]), trainable=False)
    saved_state = tf.get_variable('saved_state', initializer=tf.zeros([tower_batch_size, rnn_size]), trainable=False)

    # classifier weights and biases.
    w = tf.get_variable('Classifier_w', shape=(rnn_size, vocabulary_size), initializer=initializer)
    b = tf.get_variable('Classifier_b', shape=(vocabulary_size), initializer=initializer)

    # for the inputs
    embedded_inputs  = tf.nn.embedding_lookup(W_embedding,inputs) # tensor of shape (batch_size, seq_length, embedding_size)
    inputs_split_ = tf.split(embedded_inputs, seq_length, axis=1) # list of length seq_length with tensor elements of shape (batch_size, 1, vocabulary_size)
    list_inputs = [tf.squeeze(input_, [1]) for input_ in inputs_split_] # get rid of singleton dimensions to get list of (batch_size, vocabulary_size) tensors

    def mlstm_cell(x, h, c):
        """
        multiplicative LSTM cell. https://arxiv.org/pdf/1609.07959.pdf
        """
        # mt = (Wmxxt) ⊙ (Wmhht) - equation 18
        mt = tf.matmul(x,Wmx) * tf.matmul(h,Wmh)
        # hˆt = Whxxt + Whmmt
        ht = tf.tanh(tf.matmul(x,Whx) + tf.matmul(mt,Whm) + Whb)
        # it = σ(Wixxt + Wimmt)
        it = tf.sigmoid(tf.matmul(x,Wix) + tf.matmul(mt,Wim)+ Wib)
        # ot = σ(Woxxt + Wommt)
        ot = tf.sigmoid(tf.matmul(x,Wox) + tf.matmul(mt,Wom)+ Wob)
        # ft =σ(Wfxxt +Wfmmt)
        ft = tf.sigmoid(tf.matmul(x,Wfx) + tf.matmul(mt,Wfm)+ Wfb)

        c_new = (ft * c) + (it * ht)

        h_new = tf.tanh(c_new) * ot

        return h_new, c_new

    # Unrolled LSTM loop.
    outputs = list()
    # output and state are initially zero
    output = saved_output
    state = saved_state
    for i in list_inputs:
        output, state = mlstm_cell(i, output, state)
        outputs.append(output)

    # save the state between unrollings
    with tf.control_dependencies([saved_output.assign(output),saved_state.assign(state)]):
        # Classifier.
        logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b)   # logits.shape = (batch_size*seq_length, vocabulary_size)

    # logits is an array of shape (batch_size * seq_length, vocabulary_size). Each row is a probability mass for each input character
    return logits

def loss(logits,labels):
    """
    logits is from the inference function.

    labels is a (batch_size, seq_length) tensor
    """
    # get the labels into correct format the loss calculation
    one_hot_labels = tf.one_hot(labels, vocabulary_size)   # tensor of shape (batch_size, seq_length, vocabulary_size)
    labels_split_ = tf.split(one_hot_labels, seq_length, axis=1) # list of length seq_length with tensor elements of shape (batch_size, 1, vocabulary_size)
    list_labels = [tf.squeeze(input_, [1]) for input_ in labels_split_] # get rid of singleton dimensions to get list of (batch_size, vocabulary_size) tensors

    # calculate the loss
    labels = tf.concat(list_labels, 0)
    train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), name='train_loss')

    return train_loss

def average_gradients(tower_grads):

    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []

        # switch to indicate if gradients are sparse
        accumulate = 0

        for g, _ in grad_and_vars:

            # check if the gradients are sparse
            if re.search('embedding', g.name):
                accumulate = 1

            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, axis=0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        if accumulate == 1:
            # Sum the sparse gradients
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_sum(grad, 0)
        else:
            # Average the dense gradients
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

        # Variables are redundant because they are shared
        # across towers. Return the first tower's pointer to the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads

with tf.Graph().as_default(), tf.device('/cpu:0'):

    # Optimizer
    if args.restore_path is None:
        # for the first shard the global step begins at zero
        global_step = tf.Variable(0, name='global_step', trainable=False)
    else:
        # for subsequent shards load the global step from the last shard
        load_gs = np.load(args.restore_path + '/' + 'global_step.npy')
        global_step = tf.Variable(load_gs, name='global_step', trainable=False)

    learning_rate = tf.placeholder(tf.float32, shape=[])
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    init_lr = args.init_lr

    # lists to store the loss and gradient tensors for each tower
    tower_losses = []
    tower_grads = []

    # share the variables across each tower
    with tf.variable_scope(tf.get_variable_scope()):
        # create an instance of the model on each GPU
        for i in xrange(args.num_gpus):
            with tf.device('/gpu:%d' % i):
                # give each tower a unique namescope
                with tf.name_scope('{}_{}'.format('tower', i)) as scope:

                    # create an iterator for each tower, dataset_list contains a tf.data.Dataset object for each tower
                    iterator = dataset_list[i].make_one_shot_iterator()

                    # every time get_next() is called, the next element in the dataset is returned
                    x_batch, y_batch = iterator.get_next()

                    # compute the logits and loss for the tower
                    logits = inference(x_batch)
                    t_loss = loss(logits, y_batch)

                    # store the loss tensors to run individually if required
                    tower_losses.append(t_loss)

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Calculate the gradients for the tower
                    grads = opt.compute_gradients(t_loss, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

                    # save the gradients from each tower for the averaging
                    tower_grads.append(grads)

    # Calculate the mean of each gradient. This is the synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # track the average of the four losses during training
    average_loss = tf.reduce_mean(tower_losses, name='average_loss')

    # Apply the gradients to update the variables, which are shared across the towers
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Initializer op
    init = tf.global_variables_initializer()

    # session config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:

        # run initialization op
        sess.run(init)
        print('Variables Initialized')

        # create numpy array to store the losses during training
        logs = np.zeros(shape=(1 + num_gpus, num_steps))
        global_steps = np.zeros(shape=num_steps + 1)

        # save the initialised model before any updates
        initial_dir = os.path.join(args.saved_models_dir,'initial')
        if not os.path.exists(initial_dir):
            os.makedirs(initial_dir)
            checkpoint_path = os.path.join(initial_dir, 'model')

            tensors = tf.trainable_variables()
            weights_list = []
            for i in xrange(len(tensors)):
                a=sess.run(tensors[i])
                weights_list.append(a)
            np.save(checkpoint_path, weights_list)
            print('Initialized model saved')




        # this is the train loop
        for step in xrange(num_steps):

            # the learning rate depends on the global_step if lr_decay is selected
            gs = sess.run(global_step)
            # record the time taken for each update
            start = time.time()
            # if the lr_decay switch is on
            if args.lr_decay == 1:
                # linearly decay the learning rate to zero over the number of updates
                lr = init_lr-(((gs)*init_lr)/args.lr_decay_steps)
                if lr < 1.3*10**-14:
                    lr = 1.3*10**-14
                result = sess.run([apply_gradient_op, average_loss]  + tower_losses, feed_dict={learning_rate:lr})
                # save losses
                logs[:, step] = result[1:]
            # else use a constant learning rate throughout training
            else:
                result = sess.run([apply_gradient_op] + [average_loss] + tower_losses, feed_dict={learning_rate:init_lr})
                # save losses
                logs[:, step] = result[1:]

            duration = time.time() - start

            print("Global step: {}, progress on shard {}: ({}/{}), average_loss = {:.3f}, average_bpc = {:.3f}, time/batch = {:.3f}, learning_rate = {}"
                .format(gs, args.shard, step,num_steps, result[1], result[1]/np.log(2) ,duration, lr))

        # save the trained model weights as a list of numpy arrays
        tensors = tf.trainable_variables()
        weights_list = []
        for i in xrange(len(tensors)):
            a=sess.run(tensors[i])
            weights_list.append(a)

        # save the model weights
        model_dir = os.path.join(args.saved_models_dir,args.shard)
        os.makedirs(model_dir)
        checkpoint_path = os.path.join(model_dir, 'model')
        np.save(checkpoint_path, weights_list)
        print('Model weights saved')

        # save the global step at the end of the shard
        gs = sess.run(global_step)
        np.save(model_dir + '/' + 'global_step.npy', gs)
        print('Global step saved')

        # save the losses during training
        save_logs = os.path.join(args.log_dir, args.shard)
        os.makedirs(save_logs)
        checkpoint_path = os.path.join(save_logs, 'logs')
        np.save(checkpoint_path, logs)
        print('Logs saved')
