import os, sys, time, math, pprint, pickle, csv
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL

import numpy as np

from keras.callbacks import Callback, ModelCheckpoint, TensorBoard, EarlyStopping, LambdaCallback
from keras.layers import Dense, Dropout, Embedding, LSTM, TimeDistributed
from keras.models import load_model, Sequential
from keras.optimizers import *

import utils

def build_model(batch_size, seq_len, vocab_size=utils.VOCAB_SIZE, embedding_size=32,
                rnn_size=128, num_layers=2, drop_rate=0.0,
                learning_rate=0.001, clip_norm=5.0, optimizer='adam'):
    """
    build character embeddings LSTM text generation model.
    """
    print("building model: batch_size=%s, seq_len=%s, vocab_size=%s, "
                "embedding_size=%s, rnn_size=%s, num_layers=%s, drop_rate=%s, "
                "learning_rate=%s, clip_norm=%s.",
                batch_size, seq_len, vocab_size, embedding_size,
                rnn_size, num_layers, drop_rate,
                learning_rate, clip_norm)
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
    else: raise Exception('unsupported optimizer {}'.format(name))

def build_inference_model(model, batch_size=1, seq_len=1):
    """
    build inference model from model config
    input shape modified to (1, 1)
    """
    print("building inference model.")
    config = model.get_config()
    # edit batch_size and seq_len
    config[0]["config"]["batch_input_shape"] = (batch_size, seq_len)
    inference_model = Sequential.from_config(config)
    inference_model.trainable = False
    return inference_model


def generate_text(model, seed, length=512, top_n=10):
    """
    generates text of specified length from trained model
    with given seed character sequence.
    """
    print("generating {} characters from top {} choices.".format(length, top_n))
    print('generating with seed: "{}".'.format(seed))
    generated = seed
    encoded = utils.encode_text(seed)
    model.reset_states()

    for idx in encoded[:-1]:
        x = np.array([[idx]])
        # input shape: (1, 1)
        # set internal states
        model.predict(x)

    next_index = encoded[-1]
    for i in range(length):
        x = np.array([[next_index]])
        # input shape: (1, 1)
        probs = model.predict(x)
        # output shape: (1, 1, vocab_size)
        next_index = utils.sample_from_probs(probs.squeeze(), top_n)
        # append to sequence
        generated += utils.ID2CHAR[next_index]

    print("generated text: \n{}\n".format(generated))
    return generated

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

    val_generator = utils.io_batch_generator(train_text_path, batch_size=args['batch_size'], seq_len=args['seq_len'], one_hot_labels=True)
    train_generator = utils.io_batch_generator(val_text_path, batch_size=args['batch_size'], seq_len=args['seq_len'], one_hot_labels=True)

    train_steps_per_epoch = get_num_steps_per_epoch(train_generator)
    val_steps_per_epoch = get_num_steps_per_epoch(val_generator)
    print('train_steps_per_epoch: {}'.format(train_steps_per_epoch))
    print('val_steps_per_epoch: {}'.format(train_steps_per_epoch))

    del train_generator
    del val_generator

    val_generator = utils.io_batch_generator(train_text_path, batch_size=args['batch_size'], seq_len=args['seq_len'], one_hot_labels=True)
    train_generator = utils.io_batch_generator(val_text_path, batch_size=args['batch_size'], seq_len=args['seq_len'], one_hot_labels=True)

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

def generator_wrapper(generator):
    while True:
        x, y, _ = next(generator)
        yield x, y

def generate_main(args):
    """
    generates text from trained model specified in args.
    main method for generate subcommand.
    """
    # load learning model for config and weights
    model = load_model(args.checkpoint_path)
    # build inference model and transfer weights
    inference_model = build_inference_model(model)
    inference_model.set_weights(model.get_weights())
    print("model loaded: {}.".format(args.checkpoint_path))
    # create seed if not specified
    if args.seed is None:
        with open(args.text_path) as f:
            text = f.read()
        seed = utils.generate_seed(text)
        print("seed sequence generated from {}".format(args.text_path))
    else:
        seed = args.seed

    return generate_text(inference_model, seed, args.length, args.top_n)


def main(): 
    
    num_trials = 1
    max_epochs_per_trial = 2
    train_text_path = 'data/50M-tweets/train.txt'
    val_text_path = 'data/50M-tweets/validate.txt'
    experiment_path = 'checkpoints/base-model-50M-hyperopt'

    search_space = { 
        'batch_size': hp.choice('batch_size', [32, 64, 128, 256]),
        'drop_rate': hp.uniform('drop_rate', 0.0, 0.2),
        'embedding_size': hp.choice('embedding_size', [16, 32, 64, 128]),
        'num_layers': hp.choice('num_layers', [1, 2]),
        'rnn_size': hp.choice('rnn_size', [64, 128, 256, 512]),
        'seq_len': hp.choice('seq_len', [16, 32, 64, 128, 256]),
        'optimizer': hp.choice('optimizer', ['sgd', 
                                             'rmsprop', 
                                             'adagrad', 
                                             'adadelta', 
                                             'adam']),
        'clip_norm': hp.choice('clip_norm', [None, 5.0])
    }

    print("corpus length: {}".format(os.path.getsize(train_text_path)))
    print('vocabsize: ', utils.VOCAB_SIZE)

    trial_num = 1
    trials = []
    
    def trial(params):
        nonlocal trial_num, train_text_path, val_text_path, max_epochs_per_trial, trials
        params['num_epochs'] = max_epochs_per_trial
        checkpoint_path = '{}/{}/checkpoint.hdf5'.format(experiment_path, trial_num)

        then = time.time()
        pprint.pprint(params)

        # default values that are returned if an error is raised during the trial
        status = STATUS_OK
        error = None
        val_loss = 100
        loss = 100
        train_time = 0
        num_epochs = 0

        try:
            model, loss, val_loss, num_epochs = train(params, train_text_path, val_text_path, checkpoint_path)
        except Exception as err:
            status = STATUS_FAIL
            pudb.set_trace()
            error = err
            print(err)
        
        results = {
            'loss': val_loss,
            'status': status,
            'train_loss': loss,
            'num_epochs': num_epochs,
            'train_time': time.time() - then,
            'trial_num': trial_num,
            'error': error
        }

        trials.append([params, results])
        save_hp_checkpoint(experiment_path, trials)
        trial_num += 1
        return results

    if os.path.isdir(experiment_path):
        print('experiment_path {} already exists, exiting.'.format(experiment_path))
        exit(1)
    else: os.makedirs(experiment_path)
    fmin(fn=trial,
         space=search_space,
         algo=tpe.suggest,
         max_evals=num_trials)

    # you can test model training without running hyperparameter search like this
    # test_checkpoint = 'checkpoints/test/checkpoint.hdf5'
    # model, loss, val_loss, num_epochs = test_train(train_text_path, val_text_path, test_checkpoint)

    # past trials can be loaded like this
    # past_trials = load_trials(os.path.join(experiment_path, 'trials.pickle'))
    
    save_hp_checkpoint(experiment_path, trials)

def test_train(train_text_path, val_text_path, checkpoint_path):
    
    params = { 
        'batch_size': 64,
        'drop_rate': 0.0,
        'embedding_size': 32,
        'num_layers': 1,
        'rnn_size': 64,
        'seq_len':64,
        'optimizer': 'rmsprop',
        'clip_norm': None,
        'num_epochs': 25
    }
    
    return train(params, train_text_path, val_text_path, checkpoint_path)

def save_hp_checkpoint(experiment_path, trials):
    save_trials(os.path.join(experiment_path, 'trials.pickle'), trials)
    ranked = rank_trials(trials)
    save_trials_as_csv(os.path.join(experiment_path, 'trials.csv'), ranked)

def rank_trials(trials):
    sorted_indices = np.argsort([result['loss'] for params, result in trials])
    ranked = []
    for index in sorted_indices:
        ranked.append(trials[index])
    return ranked

def get_num_steps_per_epoch(generator):
    num_steps = 0
    while True:
        x, y, epoch = next(generator)
        if epoch > 1:
            return num_steps
        else:
            num_steps += 1 # add batch_size samples

def save_trials_as_csv(filename, ranked_trials):
    
    with open(filename, 'w') as f:
        fieldnames = ['rank', 'trial_num', 'val_loss', 'train_loss', 
                      'num_epochs', 'avg_epoch_seconds', 'batch_size', 'drop_rate',
                      'embedding_size', 'num_layers', 'rnn_size', 'seq_len', 
                      'optimizer', 'clip_norm','status']

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        rank = 1
        for trial, results in ranked_trials:
            writer.writerow({
                'rank': rank,
                'trial_num': results['trial_num'],
                'val_loss': results['loss'],
                'train_loss': results['train_loss'],
                'num_epochs': results['num_epochs'],
                'avg_epoch_seconds': int(results['train_time'] / max(results['num_epochs'], sys.float_info.epsilon)),
                'batch_size': trial['batch_size'],
                'drop_rate': trial['drop_rate'],
                'embedding_size': trial['embedding_size'],
                'num_layers': trial['num_layers'],
                'rnn_size': trial['rnn_size'],
                'seq_len': trial['seq_len'],
                'optimizer': trial['optimizer'],
                'clip_norm': trial['clip_norm'],
                'status': results['status']
            })
            rank += 1

def save_trials(filename, trials):
    with open(filename, 'wb') as f:
        pickle.dump(trials, f)

def load_trials(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
        
if __name__ == "__main__":
        main()
