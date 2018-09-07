import os
import sys
import time
import math
import pprint
import utils
import numpy as np
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard, EarlyStopping, LambdaCallback
from keras.layers import Dense, Dropout, Embedding, LSTM, TimeDistributed
from keras.models import load_model, Sequential
from keras.optimizers import *


def main():

    train_text_path = 'data/tweets-split/train.txt'
    val_text_path = 'data/tweets-split/validate.txt'
    experiment_path = 'checkpoints/tmp-delete-me'

    print("corpus length: {}".format(os.path.getsize(train_text_path)))
    print('vocabsize: ', utils.VOCAB_SIZE)

    params = {
        'batch_size': 128,
        'drop_rate': 0.0,
        'embedding_size': 64,
        'num_layers': 1,
        'rnn_size': 512,
        'seq_len': 32,
        'optimizer': 'rmsprop',
        'clip_norm': None,
        'num_epochs': 1
    }

    # you can test model training without running hyperparameter search like this
    model, loss, val_loss, num_epochs = train(params,
                                              train_text_path,
                                              val_text_path,
                                              experiment_path + '/checkpoint-2.hdf5')


def train(args, train_text_path, val_text_path, checkpoint_path):
    """
    trains model specfied in args.
    main method for train subcommand.
    """

    model = build_model(batch_size=args['batch_size'],
                        seq_len=args['seq_len'],
                        vocab_size=utils.VOCAB_SIZE,
                        embedding_size=args['embedding_size'],
                        rnn_size=args['rnn_size'],
                        num_layers=args['num_layers'],
                        drop_rate=args['drop_rate'],
                        clip_norm=args['clip_norm'],
                        optimizer=args['optimizer'])

    # make and clear checkpoint directory
    log_dir = utils.make_dirs(checkpoint_path, empty=True)
    model.save(checkpoint_path)

    print("model saved: {}.".format(checkpoint_path))
    # callbacks
    callbacks = [
        ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=3, min_delta=0.01),
        TensorBoard(os.path.join(log_dir, 'logs')),
        # you MUST reset the model's RNN states between epochs
        LambdaCallback(on_epoch_end=lambda epoch, logs: model.reset_states())
    ]

    val_generator = utils.io_batch_generator(val_text_path,
                                             batch_size=args['batch_size'],
                                             seq_len=args['seq_len'],
                                             one_hot_labels=True)
    train_generator = utils.io_batch_generator(train_text_path,
                                               batch_size=args['batch_size'],
                                               seq_len=args['seq_len'],
                                               one_hot_labels=True)

    train_steps_per_epoch = get_num_steps_per_epoch(train_generator)
    val_steps_per_epoch = get_num_steps_per_epoch(val_generator)
    print('train_steps_per_epoch: {}'.format(train_steps_per_epoch))
    print('val_steps_per_epoch: {}'.format(val_steps_per_epoch))

    del train_generator
    del val_generator

    val_generator = utils.io_batch_generator(val_text_path,
                                             batch_size=args['batch_size'],
                                             seq_len=args['seq_len'],
                                             one_hot_labels=True)
    train_generator = utils.io_batch_generator(train_text_path,
                                               batch_size=args['batch_size'],
                                               seq_len=args['seq_len'],
                                               one_hot_labels=True)

    val_generator = generator_wrapper(val_generator)
    train_generator = generator_wrapper(train_generator)

    model.reset_states()
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

    del val_generator
    del train_generator

    return model, loss, val_loss, num_epochs


def build_model(batch_size, seq_len, vocab_size=utils.VOCAB_SIZE, embedding_size=32,
                rnn_size=128, num_layers=2, drop_rate=0.0,
                learning_rate=0.001, clip_norm=5.0, optimizer='adam'):
    """
    build character embeddings LSTM text generation model.
    """
    print("building model: batch_size={}, seq_len={}, vocab_size={}, "
          "embedding_size={}, rnn_size={}, num_layers={}, drop_rate={}, "
          "learning_rate={}, clip_norm={}.".format(
              batch_size, seq_len, vocab_size, embedding_size,
              rnn_size, num_layers, drop_rate,
              learning_rate, clip_norm))
    model = Sequential()
    # input shape: (batch_size, seq_len)
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

    opt = get_optimizer(optimizer, clip_norm)
    model.compile(loss="categorical_crossentropy", optimizer=opt)
    return model


def get_optimizer(name, clip_norm):
    if name == 'sgd':
        return SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False, clipnorm=clip_norm)
    elif name == 'rmsprop':
        return RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0,  clipnorm=clip_norm)
    elif name == 'adagrad':
        return Adagrad(lr=0.01, epsilon=None, decay=0.0, clipnorm=clip_norm)
    elif name == 'adadelta':
        return Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0, clipnorm=clip_norm)
    elif name == 'adam':
        return Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipnorm=clip_norm)
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
