from loader import load_noteseqs
from model import MVAE
import argparse
import tensorflow.keras as tfk
from datetime import datetime
import shlex
from shutil import copyfile
import os
import sys
import joblib
from generate_core import MGenerator
import signal
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import kerastuner as kt
import pypianoroll
import itertools
import random

import notify_run
from dataset_stats_core import merge_dicts, get_metrics_from_midi, get_average_metrics

from tournament_core import create_model_and_train

def generate_and_save_samples(vae, epoch, path, n_genres):
    if epoch % 20 != 0:
        return

    save_path = "{}/samples/epoch-{}".format(path, epoch)

    os.makedirs(save_path, exist_ok=True)

    # TODO: Fix these magic numbers
    gen = MGenerator(21, 108, vae.x_depth, vae)

    genre_metrics = [{} for _ in range(n_genres)]

    for genre in range(n_genres):
        midis = gen.generate(genre, 50)
        metrics = {}

        for i, midi in enumerate(midis):
            midi.write("{}/genre-{}-{}.mid".format(save_path, genre, i))
            ms = get_metrics_from_midi("{}/genre-{}-{}.mid".format(save_path, genre, i))
            metrics = merge_dicts(metrics, ms)

        genre_metrics[genre] = get_average_metrics(metrics)

    series = []
    for i, stats in enumerate(genre_metrics):
        series.append(pd.Series(stats, name=str(i)))
    df_plot = pd.DataFrame(series)
    df_plot.to_csv("{}/stats.csv".format(save_path))

def parse_configuration(config):
    args = {}
    with open(config, 'r') as f:
        lines = list(line for line in (l.strip() for l in f) if line)
    for line in lines:
        k, v = shlex.split(line)  # shlex to ignore in-quote spaces
        args[k] = v
    return args

# read configuration files
config_file = sys.argv[1] if len(sys.argv) == 2 else 'train.conf'
args = parse_configuration(config_file)

x_depth = [int(d) for d in args["x_depth"].split()]
keep_pcts = [float(d) for d in args["keep_pct"].split()]
master_pct = float(args["master_pct"])
datasets = args["dataset"].split()

rnn_type = 'lstm' if args.get('rnn_type') is None else args['rnn_type']
attention = 0 if args.get('attention') is None else int(args['attention'])

print('loading datasets...')
train_segments = []
test_segments = []
for (dataset, pct) in zip(datasets, keep_pcts):
    segments = joblib.load(dataset)
    test_size = 0.1 * master_pct
    train_size = (1 - test_size) * master_pct
    Xtr, Xte = train_test_split(segments, train_size=train_size * pct, test_size=test_size * pct, random_state=42)
    train_segments.append(Xtr)
    test_segments.append(Xte)
    del segments

print('train length - test length')
for dataset, train_segment, test_segment in zip(datasets, train_segments, test_segments):
    name = dataset.split('/')[2].split('-raw.pickle')[0] 
    print(name, len(train_segment), len(test_segment))

input("Press Enter to continue...")

if int(args["cat_dim"]) != len(train_segments):
    print('{} = cat_dim != number of different datasets = {}'.format(args["cat_dim"], len(train_segments)))
    exit(1)

train_iterator = load_noteseqs(train_segments, x_depth, 64).get_iterator()
test_iterator = load_noteseqs(test_segments, x_depth, 64).get_iterator()

def create_hyper_model(hp):
    # encoder-decoder dimensions must be equal
    rnn_dim = hp.Choice('rnn_dim', [64, 128, 256, 512])
    enc_dropout = hp.Float('enc_dropout', min_value=0.2, max_value=0.7, step=0.1)
    dec_dropout = hp.Float('dec_dropout', min_value=0.2, max_value=0.7, step=0.1)
    cont_dim = hp.Choice('cont_dim', [50, 120, 170, 250])
    mu_force = hp.Choice('mu_force', [0.5, 1.2, 1.8, 5.0])
    t_gumbel = hp.Choice('t_gumbel', [0.001, 0.02, 0.1])
    style_dim = hp.Choice('style_dim', [20, 80, 150])
    kl_reg = hp.Choice('kl_reg', [0.2, 0.8, 1.5])
    beta_steps = hp.Choice('beta_anneal_steps', [500, 1000, 1800, 3000])
    attention = hp.Choice('attention', [0, 128, 256, 512])
    lr = hp.Choice('lr', [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4])

    vae = MVAE(x_depth=x_depth,
                enc_rnn_dim=rnn_dim, enc_dropout=enc_dropout,
                dec_rnn_dim=rnn_dim, dec_dropout=dec_dropout,
                cont_dim=cont_dim, cat_dim=int(args["cat_dim"]), mu_force=mu_force,
                t_gumbel=t_gumbel, style_embed_dim=style_dim,
                kl_reg=kl_reg,
                beta_anneal_steps=beta_steps,
                rnn_type="lstm", attention=attention)

    optimizer = tfk.optimizers.Adam(learning_rate=lr)
    vae.compile(optimizer=optimizer)
    vae.run_eagerly = True

    return vae

tuner = kt.Hyperband(create_hyper_model,
                     objective=kt.Objective('val_p_acc', direction='max'),
                     max_epochs=5,
                     factor=3,
                     directory='tournament',
                     project_name='hyperparam-tuning')  

print(tuner.search_space_summary())
input("Begin? (Press enter)...")

callbacks = [
    tfk.callbacks.LambdaCallback(on_epoch_start=lambda epoch,logs: logs['model'].reset_trackers()),
    tfk.callbacks.EarlyStopping(monitor='val_p_acc', min_delta=0.01, patience=5, mode='max'),
    # tfk.callbacks.CSVLogger('tournament/log.csv', append=True),    

    # tfk.callbacks.TensorBoard(log_dir="tournament/logs", write_graph=True, update_freq='epoch')
]


tuner.search(train_iterator, callbacks=callbacks, validation_data=test_iterator)

best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

best_dict = {
    'enc_rnn_dim': best_hps.get('rnn_dim'), 
    'dec_rnn_dim': best_hps.get('rnn_dim'), 
    'enc_dropout': best_hps.get('enc_dropout'), 
    'dec_dropout': best_hps.get('dec_dropout'), 
    'cont_dim': best_hps.get('cont_dim'), 
    'mu_force': best_hps.get('mu_force'), 
    't_gumbel': best_hps.get('t_gumbel'), 
    'style_dim': best_hps.get('style_dim'), 
    'kl_reg': best_hps.get('kl_reg'), 
    'beta_anneal_steps': best_hps.get('beta_anneal_steps'), 
    'attention': best_hps.get('attention'),
    'lr': best_hps.get('lr')
}

print("Best hyperparameter configuration")
print(best_dict)

best_model = tuner.hypermodel.build(best_hps)

callbacks = [
    tfk.callbacks.LambdaCallback(on_epoch_end=lambda epoch,_: generate_and_save_samples(best_model, epoch, 'tournament', 2)),
    tfk.callbacks.LambdaCallback(on_epoch_start=lambda epoch,_: best_model.reset_trackers()),

    tfk.callbacks.CSVLogger('tournament/log.csv', append=True),    
    tfk.callbacks.ModelCheckpoint('tournament/weights/' + '/weights.{epoch:02d}', monitor='val_p_acc', save_weights_only=True, save_best_only=True, mode='max'),
    tfk.callbacks.TensorBoard(log_dir='tournament', update_freq='epoch', histogram_freq=40, profile_batch='10,20')
]

print("Beginning training the best model...")

best_model.load_weights('tournament/weights/weights.294')

best_model.fit(train_iterator, epochs=500, initial_epoch=300, callbacks=callbacks, validation_data=test_iterator)

# notifier = notify_run.Notify()
# notifier.send(str(best_dict))