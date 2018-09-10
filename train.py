import os
import sys
import time
import math
import pprint
import utils
import numpy as np
from argparse import ArgumentParser
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard, EarlyStopping, LambdaCallback, LearningRateScheduler
from keras.layers import Dense, Dropout, Embedding, LSTM, TimeDistributed
from keras.models import load_model, Sequential
from keras.optimizers import *


def main():

    args = parse_args()

    # we'll use the train.txt and validate.txt files from the  --data-dir folder
    # specified in the command line arguments (data/twitter-split by default)
    train_text_path = os.path.join(args.data_dir, 'train.txt')
    val_text_path = os.path.join(args.data_dir, 'validate.txt')

    if not os.path.exists(os.path.join(args.data_dir, 'train.txt')):
        print('train.txt does not exist in --data-dir: {}'.format(args.data_dir))
        exit(1)

    if not os.path.exists(os.path.join(args.data_dir, 'validate.txt')):
        print('validate.txt does not exist in --data-dir: {}'.format(args.data_dir))
        exit(1)

    if not args.restore and os.path.isdir(args.checkpoint_dir):
        print('--checkpoint_dir {} already exits and --restore flag is not present.'.format(args.checkpoint_dir))
        exit(1)

    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    print('loading training data from {}'.format(train_text_path))
    print('corpus length: {}'.format(os.path.getsize(train_text_path)))
    print('vocabsize: ', utils.VOCAB_SIZE)

    # here is an example of the values returned by train(...). We don't use them
    # here, but you can if you want to modify something in the future:
    #     - model: the trained model
    #     - loss: the final training loss value
    #     - val_loss: the final validation loss value
    #     - num_epochs: the number of epochs the model was trained for
    model, loss, val_loss, num_epochs = train(vars(args),
                                              train_text_path,
                                              val_text_path)


def parse_args():

    arg_parser = ArgumentParser(
        description="train an LSTM text generation model")
    arg_parser.add_argument("--checkpoint-dir", required=True,
                            help="path to save or load model checkpoints (required)")
    arg_parser.add_argument("--data-dir", default="data/tweets-split",
                            help="path to a directory containing a train.txt and validate.txt file (required)")
    arg_parser.add_argument("--restore", action='store_true',
                            help="restore training from a checkpoint.hdf5 file in --checkpoint-dir.")
    arg_parser.add_argument("--num-layers", type=int, default=1,
                            help="number of rnn layers (default: %(default)s)")
    arg_parser.add_argument("--rnn-size", type=int, default=512,
                            help="size of rnn cell (default: %(default)s)")
    arg_parser.add_argument("--embedding-size", type=int, default=64,
                            help="character embedding size (default: %(default)s)")
    arg_parser.add_argument("--batch-size", type=int, default=128,
                            help="training batch size (default: %(default)s)")
    arg_parser.add_argument("--seq-len", type=int, default=32,
                            help="sequence length of inputs and outputs (default: %(default)s)")
    arg_parser.add_argument("--drop-rate", type=float, default=0.05,
                            help="dropout rate for rnn layers (default: %(default)s)")
    arg_parser.add_argument("--learning-rate", type=float, default=None,
                            help="learning rate (default: the default keras learning rate for the chosen optimizer)")
    arg_parser.add_argument("--clip-norm", type=float, default=5.0,
                            help="max norm to clip gradient (default: %(default)s)")
    arg_parser.add_argument("--optimizer", type=str, default='rmsprop',
                            choices=['sgd', 'rmsprop',
                                     'adagrad', 'adadelta', 'adam'],
                            help="optimizer name (default: %(default)s)")
    arg_parser.add_argument("--num-epochs", type=int, default=10,
                            help="number of epochs for training (default: %(default)s)")
    return arg_parser.parse_args()


def train(args, train_text_path, val_text_path):
    """
    trains model using hyperparameters specified in args.
    """

    #
    checkpoint_path = os.path.join(args['checkpoint_dir'], 'checkpoint.hdf5')

    model = None
    # restore the model from checkpoint.hdf5 file in --checkpoint-dir if --restore
    # was specified in args
    if args['restore']:
        if not os.path.exists(checkpoint_path):
            err = 'cannot restore model from a checkpoint path that doesn\'t exist: {}'.format(
                checkpoint_path)
            raise Exception(err)
        model = load_model(checkpoint_path)
        print('model loaded from {}'.format(checkpoint_path))
    # otherwise, we will create a new model from scratch...
    else:
        model = build_model(batch_size=args['batch_size'],
                            seq_len=args['seq_len'],
                            vocab_size=utils.VOCAB_SIZE,
                            embedding_size=args['embedding_size'],
                            rnn_size=args['rnn_size'],
                            num_layers=args['num_layers'],
                            drop_rate=args['drop_rate'],
                            clip_norm=args['clip_norm'],
                            optimizer=args['optimizer'])

    # once we've loaded/built the model, we need to compile it using an 
    # optimizer and a loss. The optimizer can be one of several strings (see get_optimizer())
    # but the loss must be categorical_crossentropy, because our task is 
    # a multi-class classification problem.
    opt = get_optimizer(args['optimizer'],
                        args['clip_norm'], args['learning_rate'])
    model.compile(loss="categorical_crossentropy", optimizer=opt)

    # Callbacks are hooked functions that are keras will call automagically
    # during model training. For more info, see https://keras.io/callbacks/
    #     - EarlyStopping: Will stop model training once val_loss plateuas
    #     - TensorBoard: logs training metrics so that they may be viewed by tensorboard
    #     - ModelCheckpoint: save model checkpoints after each epoch. Only save a checkpoint if val_loss has improved.
    #     - LabmdaCallback: Your own custom hooked function. Here we reset the model's RNN states between Epochs
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, min_delta=0.01),
        TensorBoard(os.path.join(args['checkpoint_dir'], 'logs')),
        ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True),
        # you MUST reset the model's RNN states between epochs
        # LearningRateScheduler(lr_schedule, verbose=1),
        LambdaCallback(on_epoch_end=lambda epoch, logs: model.reset_states())
    ]

    if not args['restore']:
        model.save(checkpoint_path)

    # because we may not be able to fit all of our training data into RAM at once
    # we will use a python generator that lazy-loads (and releases) data from disk
    # into RAM as we need it. We will use one generator for validation data
    # and another for training data.
    val_generator = utils.io_batch_generator(val_text_path,
                                             batch_size=args['batch_size'],
                                             seq_len=args['seq_len'],
                                             one_hot_labels=True)
    train_generator = utils.io_batch_generator(train_text_path,
                                               batch_size=args['batch_size'],
                                               seq_len=args['seq_len'],
                                               one_hot_labels=True)

    # the way the generator is written, we won't know how many samples are in
    # the entire dataset without processing it all once. We need to know this
    # number so that we can know the number of batch steps per epoch. This isn't elegant,
    # but it is a tradeoff that is worth making to have no limit to the amount of
    # training data we can process using our generator.
    train_steps_per_epoch = get_num_steps_per_epoch(train_generator)
    val_steps_per_epoch = get_num_steps_per_epoch(val_generator)
    print('train_steps_per_epoch: {}'.format(train_steps_per_epoch))
    print('val_steps_per_epoch: {}'.format(val_steps_per_epoch))

    # now that we've computed train_steps_per_epoch and val_steps_per_epoch
    # we will re-create the generators so that they begin at epoch 0 instead of 1
    val_generator = utils.io_batch_generator(val_text_path,
                                             batch_size=args['batch_size'],
                                             seq_len=args['seq_len'],
                                             one_hot_labels=True)
    train_generator = utils.io_batch_generator(train_text_path,
                                               batch_size=args['batch_size'],
                                               seq_len=args['seq_len'],
                                               one_hot_labels=True)

    # our io_batch_generator returns (x, y, epoch) but model.fit_generator()
    # expects only (x, y) so we create generator_wrapper() which simply disposes
    # of the extra "epoch" tuple element.
    val_generator = generator_wrapper(val_generator)
    train_generator = generator_wrapper(train_generator)

    model.reset_states()

    # train the model using train_generator for training data and val_generator
    # for validation data.
    history = model.fit_generator(train_generator,
                                  epochs=args['num_epochs'],
                                  steps_per_epoch=train_steps_per_epoch,
                                  validation_data=val_generator,
                                  validation_steps=val_steps_per_epoch,
                                  callbacks=callbacks)

    pprint.pprint(history.history)
    loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    num_epochs = len(history.history['loss'])

    return model, loss, val_loss, num_epochs


# def lr_schedule(epoch):
#     if epoch <= 8:
#         return 0.001
#     elif epoch <= 15:
#         return 0.0005
#     else:
#         return 0.00025


def build_model(batch_size, seq_len, vocab_size=utils.VOCAB_SIZE,
                embedding_size=32, rnn_size=128, num_layers=2, drop_rate=0.0):
    """
    build character embeddings LSTM text generation model.
    """
    print("building model: batch_size={}, seq_len={}, vocab_size={}, "
          "embedding_size={}, rnn_size={}, num_layers={}, drop_rate={}, "
          "learning_rate={}, clip_norm={}.".format(
              batch_size, seq_len, vocab_size, embedding_size,
              rnn_size, num_layers, drop_rate, clip_norm))

    model = Sequential()
    # input shape: (batch_size, seq_len)
    # add an embedding layer to map input indexes to fixed-width learned vectors
    model.add(Embedding(vocab_size, embedding_size,
                        batch_input_shape=(batch_size, seq_len)))
    model.add(Dropout(drop_rate))
    # shape: (batch_size, seq_len, embedding_size)
    for _ in range(num_layers):
        model.add(LSTM(rnn_size, return_sequences=True, stateful=True))
        model.add(Dropout(drop_rate))
    # shape: (batch_size, seq_len, rnn_size)
    model.add(TimeDistributed(Dense(vocab_size, activation="softmax")))
    # output shape: (batch_size, seq_len, vocab_size)
    return model


def get_optimizer(name, clip_norm, learning_rate=None):
    if name == 'sgd':
        lr = 0.01 if learning_rate == None else learning_rate
        return SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False, clipnorm=clip_norm)
    elif name == 'rmsprop':
        lr = 0.001 if learning_rate == None else learning_rate
        return RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0,  clipnorm=clip_norm)
    elif name == 'adagrad':
        lr = 0.01 if learning_rate == None else learning_rate
        return Adagrad(lr=lr, epsilon=None, decay=0.0, clipnorm=clip_norm)
    elif name == 'adadelta':
        lr = 1.0 if learning_rate == None else learning_rate
        return Adadelta(lr=lr, rho=0.95, epsilon=None, decay=0.0, clipnorm=clip_norm)
    elif name == 'adam':
        lr = 0.001 if learning_rate == None else learning_rate
        return Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipnorm=clip_norm)
    else:
        raise Exception('unsupported optimizer {}'.format(name))


def generator_wrapper(generator):
    while True:
        x, y, _ = next(generator)
        yield x, y


def get_num_steps_per_epoch(generator):
    num_steps = 0
    while True:
        x, y, epoch = next(generator)
        if epoch > 1:
            return num_steps
        else:
            num_steps += 1  # add batch_size samples


if __name__ == "__main__":
    main()
