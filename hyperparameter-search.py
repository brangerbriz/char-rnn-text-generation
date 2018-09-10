import os
import sys
import time
import pprint
import pickle
import csv
import utils
import train
import numpy as np
from hyperopt import fmin, tpe, rand, hp, STATUS_OK, STATUS_FAIL

# the number of individual models to train using different hyperparameters
NUM_TRIALS = 20
# the maximum number of epochs per trial
MAX_EPOCHS_PER_TRIAL = 10
TRAIN_TEXT_PATH = 'data/tweets-split-tmp/train.txt'
VAL_TEXT_PATH = 'data/tweets-split-tmp/validate.txt'
# trials will be saved in this directory in separate folders specified by their
# trial number (e.g. 1/, 2/, 3/, 4/, etc.)
EXPERIMENT_PATH = 'checkpoints/20-trials-10-epochs'

# each trial will sample values from this search space to train a new model.
# see hyperopt's documentation if you would like to add different types of 
# sampling configurations.
SEARCH_SPACE = {
    'batch_size': hp.choice('batch_size', [16, 32, 64, 128, 256, 512]),
    'drop_rate': 0.0,
    'embedding_size': hp.choice('embedding_size', [16, 32, 64, 128, 256]),
    'num_layers': 1, # you can replace these constants with hp.choice() or hp.uniform(), etc.
    'rnn_size': 512,
    'seq_len': hp.choice('seq_len', [16, 32, 64, 128, 256]),
    'optimizer': hp.choice('optimizer', ['rmsprop',
                                         'adagrad',
                                         'adadelta',
                                         'adam']),
    'clip_norm': hp.choice('clip_norm', [0.0, 5.0])
}

# Use "Tree of Parzen Estimators" as the search algorithm by default. 
# You can switch to "Random Search" instead with:
#     SEARCH_ALGORITHM=rand.suggest
SEARCH_ALGORITHM=tpe.suggest

def main():

    def trial(params):
        global TRAIN_TEXT_PATH, VAL_TEXT_PATH, MAX_EPOCHS_PER_TRIAL
        nonlocal trial_num, trials
        params['num_epochs'] = MAX_EPOCHS_PER_TRIAL
        params['checkpoint_dir'] = '{}/{}/'.format(EXPERIMENT_PATH, trial_num)
        os.makedirs(params['checkpoint_dir'])

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
            model, loss, val_loss, num_epochs = train.train(params, 
                                                            TRAIN_TEXT_PATH, 
                                                            VAL_TEXT_PATH)
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
         algo=SEARCH_ALGORITHM,
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
