import os
import sys
import time
import pprint
import pickle
import csv
import utils
import train
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL

NUM_TRIALS = 100
MAX_EPOCHS_PER_TRIAL = 1
TRAIN_TEXT_PATH = 'data/tweets-split-tmp/train.txt'
VAL_TEXT_PATH = 'data/tweets-split-tmp/validate.txt'
EXPERIMENT_PATH = 'checkpoints/tmp-delete-me'

SEARCH_SPACE = {
    'batch_size': hp.choice('batch_size', [32, 64, 128, 256]),
    'drop_rate': 0.0,
    'embedding_size': hp.choice('embedding_size', [16, 32, 64, 128, 256]),
    'num_layers': 1,
    'rnn_size': hp.choice('rnn_size', [256, 512, 1024]),
    'seq_len': hp.choice('seq_len', [16, 32, 64, 128, 256]),
    'optimizer': hp.choice('optimizer', ['sgd',
                                         'rmsprop',
                                         'adagrad',
                                         'adadelta',
                                         'adam']),
    'clip_norm': hp.choice('clip_norm', [0.0, 5.0])
}


def main():

    def trial(params):
        global TRAIN_TEXT_PATH, VAL_TEXT_PATH, MAX_EPOCHS_PER_TRIAL
        nonlocal trial_num, trials
        params['num_epochs'] = MAX_EPOCHS_PER_TRIAL
        checkpoint_path = '{}/{}/checkpoint.hdf5'.format(
            EXPERIMENT_PATH, trial_num)

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
            model, loss, val_loss, num_epochs = train.train(
                params, TRAIN_TEXT_PATH, VAL_TEXT_PATH, checkpoint_path)
        except Exception as err:
            status = STATUS_FAIL
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
        save_hp_checkpoint(EXPERIMENT_PATH, trials)
        trial_num += 1
        return results

    print("corpus length: {}".format(os.path.getsize(TRAIN_TEXT_PATH)))
    print('vocabsize: ', utils.VOCAB_SIZE)

    trial_num = 1
    trials = []

    if os.path.isdir(EXPERIMENT_PATH):
        print('EXPERIMENT_PATH {} already exists, exiting.'.format(EXPERIMENT_PATH))
        exit(1)
    else:
        os.makedirs(EXPERIMENT_PATH)

    # run the hyperparameter search
    fmin(fn=trial,
         space=SEARCH_SPACE,
         algo=tpe.suggest,
         max_evals=NUM_TRIALS)

    # past trials can be loaded like this
    # past_trials = load_trials(os.path.join(EXPERIMENT_PATH, 'trials.pickle'))

    save_hp_checkpoint(EXPERIMENT_PATH, trials)


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


def save_trials_as_csv(filename, ranked_trials):
    with open(filename, 'w') as f:
        fieldnames = ['rank', 'trial_num', 'val_loss', 'train_loss',
                      'num_epochs', 'avg_epoch_seconds', 'batch_size', 'drop_rate',
                      'embedding_size', 'num_layers', 'rnn_size', 'seq_len',
                      'optimizer', 'clip_norm', 'status']

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
