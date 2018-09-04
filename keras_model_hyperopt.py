import os
import time, math, pprint, pickle
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL

import numpy as np

from keras.callbacks import Callback, ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import Dense, Dropout, Embedding, LSTM, TimeDistributed
from keras.models import load_model, Sequential
from keras.optimizers import Adam

from logger import get_logger
import utils

logger = get_logger(__name__)


def build_model(batch_size, seq_len, vocab_size=utils.VOCAB_SIZE, embedding_size=32,
                rnn_size=128, num_layers=2, drop_rate=0.0,
                learning_rate=0.001, clip_norm=5.0):
    """
    build character embeddings LSTM text generation model.
    """
    logger.info("building model: batch_size=%s, seq_len=%s, vocab_size=%s, "
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
    optimizer = Adam(learning_rate, clipnorm=clip_norm)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    return model


def build_inference_model(model, batch_size=1, seq_len=1):
    """
    build inference model from model config
    input shape modified to (1, 1)
    """
    logger.info("building inference model.")
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
    logger.info("generating %s characters from top %s choices.", length, top_n)
    logger.info('generating with seed: "%s".', seed)
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

    logger.info("generated text: \n%s\n", generated)
    return generated


class LoggerCallback(Callback):
    """
    callback to log information.
    generates text at the end of each epoch.
    """
    def __init__(self, text, model):
        super(LoggerCallback, self).__init__()
        self.text = text
        # build inference model using config from learning model
        self.inference_model = build_inference_model(model)
        self.time_train = self.time_epoch = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.time_epoch = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration_epoch = time.time() - self.time_epoch
        logger.info("epoch: %s, duration: %ds, loss: %.6g., val_loss: %.6g",
                    epoch, duration_epoch, logs["loss"], logs["val_loss"])
        # transfer weights from learning model
        self.inference_model.set_weights(self.model.get_weights())

        # generate text
        seed = utils.generate_seed(self.text)
        generate_text(self.inference_model, seed)

    def on_train_begin(self, logs=None):
        logger.info("start of training.")
        self.time_train = time.time()

    def on_train_end(self, logs=None):
        duration_train = time.time() - self.time_train
        logger.info("end of training, duration: %ds.", duration_train)
        # transfer weights from learning model
        self.inference_model.set_weights(self.model.get_weights())

        # generate text
        seed = utils.generate_seed(self.text)
        generate_text(self.inference_model, seed, 1024, 3)


def train_main(args, text):

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
                        learning_rate=args['learning_rate'],
                        clip_norm=args['clip_norm'])

    # make and clear checkpoint directory
    log_dir = utils.make_dirs(args['checkpoint_path'], empty=True)
    model.save(args['checkpoint_path'])

    logger.info("model saved: %s.", args['checkpoint_path'])
    # callbacks
    callbacks = [
        ModelCheckpoint(args['checkpoint_path'], verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=3),
        TensorBoard(os.path.join(log_dir, 'logs')),
        LoggerCallback(text, model)
    ]


    val_split = 0.2
    val_split_index = math.floor(len(text) * val_split)
    # training start
    num_batches = (len(text) - val_split_index - 1) // (args['batch_size'] * args['seq_len'])
    num_val_batches = val_split_index // (args['batch_size'] * args['seq_len'])
    print('{} num batches'.format(num_batches))
    print('{} num val batches'.format(num_val_batches))

    val_generator = utils.batch_generator(utils.encode_text(text[0:val_split_index]), args['batch_size'], args['seq_len'], one_hot_labels=True)
    train_generator = utils.batch_generator(utils.encode_text(text[val_split_index:]), args['batch_size'], args['seq_len'], one_hot_labels=True)
    model.reset_states()
    x, y = next(train_generator) 
    history = model.fit_generator(train_generator,
                        num_batches,
                        args['num_epochs'],
                        validation_data=val_generator,
                        validation_steps=num_val_batches,
                        callbacks=callbacks)
    pprint.pprint(history.history)
    loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    num_epochs = len(history.history['loss'])
    return model, loss, val_loss, num_epochs


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
    logger.info("model loaded: %s.", args.checkpoint_path)
    # create seed if not specified
    if args.seed is None:
        with open(args.text_path) as f:
            text = f.read()
        seed = utils.generate_seed(text)
        logger.info("seed sequence generated from %s.", args.text_path)
    else:
        seed = args.seed

    return generate_text(inference_model, seed, args.length, args.top_n)


def main(): 
    
    num_trials = 3
    num_epochs_per_trial = 10
    text_path = 'data/10M-tweets.txt'
    experiment_path = 'checkpoints/base-model-10M-hyperopt'

    choices = {
        'batch_size': [64, 128 , 256],
        'embedding_size': [16, 32, 64, 128],
        'num_layers': [1, 2, 3],
        'rnn_size': [64, 128, 256, 512],
        'seq_len': [16, 32, 64, 128, 256]
    }
    
    search_space = { 
        'batch_size': hp.choice('batch_size', choices['batch_size']),
        'drop_rate': hp.uniform('drop_rate', 0.0, 0.3),
        'embedding_size': hp.choice('embedding_size', choices['embedding_size']),
        'num_layers': hp.choice('num_layers', choices['num_layers']),
        'rnn_size': hp.choice('rnn_size', choices['rnn_size']),
        'seq_len': hp.choice('seq_len', choices['seq_len']),
        'learning_rate': 0.001
    }

    # load text
    with open(text_path) as f:
        text = f.read()[0:100000]

    logger.info("corpus length: %s.", len(text))
    print('vocabsize: ', utils.VOCAB_SIZE)

    trial_num = 1
    my_trials = []
    
    def trial(params):
        nonlocal trial_num, text, num_epochs_per_trial, my_trials
        params['checkpoint_path'] = '{}/{}/checkpoint.hdf5'.format(experiment_path, trial_num)
        params['text_path'] = text_path
        params['num_epochs'] = num_epochs_per_trial
        params['clip_norm'] = 5.0
        params['trial_num'] = trial_num

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
            model, loss, val_loss, num_epochs = train_main(params, text)
        except Exception as err:
            status = STATUS_FAIL
            error = err
            print(err)
        
        trial_num += 1
        results = {
            'loss': val_loss,
            'status': status,
            'training_loss': loss,
            'num_epochs': num_epochs,
            'train_time': time.time() - then,
            'error': error
        }

        my_trials.append([params, results])

        return results

    # trials = Trials()
    # best = fmin(fn=trial,
    #             space=search_space,
    #             algo=tpe.suggest,
    #             max_evals=num_trials,
    #             trials=trials)
    
    # save_trials(os.path.join(experiment_path, 'trials'), trials, best, choices, my_trials)

    past_trials = load_trials(os.path.join(experiment_path, 'trials'))
    print(past_trials['my_trials'])
    ranked = get_ranked_hyperparameters(past_trials['my_trials'])
    pprint.pprint(ranked)

    # assert(best['drop_rate'] == )

def save_trials(_dir, trials, best, choices, my_trials):

    os.makedirs(_dir)

    with open(os.path.join(_dir, 'trials.pickle'), 'wb') as f:
        pickle.dump(trials.trials, f)

    with open(os.path.join(_dir, 'results.pickle'), 'wb') as f:
        pickle.dump(trials.results, f)

    with open(os.path.join(_dir, 'losses.pickle'), 'wb') as f:
        pickle.dump(trials.losses(), f)
    
    with open(os.path.join(_dir, 'best.pickle'), 'wb') as f:
        pickle.dump(best, f)

    with open(os.path.join(_dir, 'choices.pickle'), 'wb') as f:
        pickle.dump(choices, f)

    with open(os.path.join(_dir, 'my_trials.pickle'), 'wb') as f:
        pickle.dump(my_trials, f)

def load_trials(_dir):

    with open(os.path.join(_dir, 'trials.pickle'), 'rb') as f:
        trials = pickle.load(f)

    with open(os.path.join(_dir, 'results.pickle'), 'rb') as f:
        results = pickle.load(f)

    with open(os.path.join(_dir, 'losses.pickle'), 'rb') as f:
        losses = pickle.load(f)

    with open(os.path.join(_dir, 'best.pickle'), 'rb') as f:
        best = pickle.load(f)

    with open(os.path.join(_dir, 'choices.pickle'), 'rb') as f:
        choices = pickle.load(f)

    with open(os.path.join(_dir, 'my_trials.pickle'), 'rb') as f:
        my_trials = pickle.load(f)

    return {
        'trials': trials,
        'results': results,
        'losses': losses,
        'best': best,
        'choices': choices,
        'my_trials': my_trials
    }

def get_ranked_hyperparameters(my_trials):
    sorted_indices = np.argsort([result['loss'] for params, result in my_trials])
    ranked = []
    trial_num = 1
    for index in sorted_indices:
        trial, result = my_trials[index]
        ranked.append((result['loss'], trial))
        # print(index)
        # ranked.append([my_trials[index][1]['loss'], trial_num, my_trials[index][0]])
        trial_num += 1
    print(ranked)
    return ranked
    # for index in sorted_indices:
    # [[val_loss, loss, experiment_number, parameters]]

def get_hyperparameter_values(hyperopt_obj, choices):
    pass

if __name__ == "__main__":
    main()
