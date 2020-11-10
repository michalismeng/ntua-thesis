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
import tensorflow as tf
import gc

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
    path = "{}/samples".format(path)
    df_plot.to_csv("{}/stats.csv".format(path), mode='a', header=(epoch < 5))

def parse_configuration(config):
    args = {}
    with open(config, 'r') as f:
        lines = list(line for line in (l.strip() for l in f) if line)
    for line in lines:
        k, v = shlex.split(line)  # shlex to ignore in-quote spaces
        args[k] = v
    return args

def get_dict_from_hps(hp):
    return {
        'enc_rnn_dim': hp.get('rnn_dim'), 
        'dec_rnn_dim': hp.get('rnn_dim'), 
        'enc_dropout': hp.get('enc_dropout'), 
        'dec_dropout': hp.get('dec_dropout'), 
        'cont_dim': hp.get('cont_dim'), 
        't_gumbel': hp.get('t_gumbel'), 
        'style_dim': hp.get('style_dim'), 
        'kl_reg': hp.get('kl_reg'), 
        'beta_anneal_steps': hp.get('beta_anneal_steps'), 
        'attention': hp.get('attention'),
        'lr': hp.get('lr'),
        'decay': hp.get('decay'),
        'decay_steps': hp.get('decay_steps'),
        'rnn_type': hp.get('rnn_type')
    }

# read configuration files
config_file = sys.argv[1] if len(sys.argv) == 2 else 'train.conf'
args = parse_configuration(config_file)

x_depth = [int(d) for d in args["x_depth"].split()]
keep_pcts = [float(d) for d in args["keep_pct"].split()]
master_pct = float(args["master_pct"])
datasets = args["dataset"].split()
cat_dim = int(args["cat_dim"])
batch_size = int(args["batch_size"])

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

if cat_dim != len(train_segments):
    print('{} = cat_dim != number of different datasets = {}'.format(args["cat_dim"], len(train_segments)))
    exit(1)

train_iterator = load_noteseqs(train_segments, x_depth, batch_size).get_iterator()
test_iterator = load_noteseqs(test_segments, x_depth, batch_size).get_iterator()

count_models = 0

def create_hyper_model(hp):
    global count_models
    # reset keras global state (hopefully releasing memory)
    tf.keras.backend.clear_session()
    gc.collect()

    count_models = count_models + 1

    # encoder-decoder dimensions must be equal
    rnn_dim = hp.Choice('rnn_dim', [512])
    enc_dropout = hp.Float('enc_dropout', min_value=0.4, max_value=0.7, step=0.1)
    dec_dropout = hp.Float('dec_dropout', min_value=0.4, max_value=0.7, step=0.1)
    cont_dim = hp.Choice('cont_dim', [20, 50, 120, 200, 400])
    mu_force = 1.2 #hp.Choice('mu_force', [0.5, 1.2, 2.5, 5.0])
    t_gumbel = hp.Choice('t_gumbel', [0.0005, 0.001, 0.02, 0.1])
    style_dim = hp.Choice('style_dim', [20, 80, 150, 300])
    kl_reg = hp.Choice('kl_reg', [0.2, 0.5, 0.8, 0.9])
    beta_steps = hp.Choice('beta_anneal_steps', [2500, 5000])
    attention = hp.Choice('attention', [0, 128, 256, 512])
    lr = hp.Choice('lr', [5e-4, 5.5e-4, 6e-4, 8e-4, 1e-3])
    decay = hp.Choice('decay', [0.85, 0.93, 0.95, 0.97])
    decay_steps = hp.Choice('decay_steps', [2500])
    rnn_type = hp.Choice('rnn_type', ["lstm", "gru"])

    # rnn_dim = hp.Choice('rnn_dim', [512])
    # enc_dropout = hp.Choice('enc_dropout', [0.5])
    # dec_dropout = hp.Choice('dec_dropout', [0.2])
    # cont_dim = hp.Choice('cont_dim', [120])
    # mu_force = 1.3
    # t_gumbel = hp.Choice('t_gumbel', [0.02])
    # style_dim = hp.Choice('style_dim', [80])
    # kl_reg = hp.Choice('kl_reg', [0.8])
    # beta_steps = hp.Choice('beta_anneal_steps', [2500])
    # attention = hp.Choice('attention', [128])
    # lr = hp.Choice('lr', [5e-4])
    # decay = hp.Choice('decay', [0.85])
    # decay_steps = hp.Choice('decay_steps', [2500])
    # rnn_type = hp.Choice('rnn_type', ["lstm"])

    vae = MVAE(x_depth=x_depth,
                enc_rnn_dim=rnn_dim, enc_dropout=enc_dropout,
                dec_rnn_dim=rnn_dim, dec_dropout=dec_dropout,
                cont_dim=cont_dim, cat_dim=cat_dim, mu_force=mu_force,
                t_gumbel=t_gumbel, style_embed_dim=style_dim,
                kl_reg=kl_reg,
                beta_anneal_steps=beta_steps,
                rnn_type=rnn_type, attention=attention)

    schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps, decay, staircase=False)

    optimizer = tfk.optimizers.Adam(learning_rate=schedule)
    vae.compile(optimizer=optimizer)
    vae.run_eagerly = True

    # enable annealing
    vae.set_kl_anneal(True)

    return vae

directory = 'final-jsb-nmd-8'
tuner = kt.Hyperband(create_hyper_model,
                     objective=kt.Objective('val_p_acc', direction='max'),
                     max_epochs=250,
                     hyperband_iterations=2,
                     factor=7,
                     directory=directory,
                     project_name='hyperparam-tuning')  

print(tuner.search_space_summary())
input("Begin? (Press enter)...")

callbacks = [
    tfk.callbacks.LambdaCallback(on_epoch_start=lambda epoch,logs: logs['model'].reset_trackers()),
    tfk.callbacks.EarlyStopping(monitor='val_p_acc', min_delta=0.01, patience=20, mode='max'),
]

tuner.search(train_iterator, callbacks=callbacks, validation_data=test_iterator)

top_10_hps = tuner.get_best_hyperparameters(num_trials = 10)
best_hps = top_10_hps[0]
best_dict = get_dict_from_hps(best_hps)

print("Best hyperparameter configuration")
print(best_dict)

with open(f'{directory}/best-config.txt', 'w') as f:
    f.write(str(best_dict))
    f.write("\n\n")
    f.write(f"Checked {count_models} models.")

with open(f'{directory}/top-10-configs.txt', 'w') as f:
    for hp in top_10_hps:
        f.write(str(get_dict_from_hps(hp)))
        f.write("\n\n")

# notifier = notify_run.Notify()
# notifier.send(str(best_dict))

best_model = tuner.hypermodel.build(best_hps)

best_model.set_kl_anneal(True)

callbacks = [
    tfk.callbacks.LambdaCallback(on_epoch_end=lambda epoch,_: generate_and_save_samples(best_model, epoch, f'{directory}', cat_dim)),
    tfk.callbacks.LambdaCallback(on_epoch_start=lambda epoch,_: best_model.reset_trackers()),

    tfk.callbacks.CSVLogger(f'{directory}/log.csv', append=True),    
    tfk.callbacks.ModelCheckpoint(f'{directory}/weights' + '/weights.{epoch:02d}', monitor='val_p_acc', save_weights_only=True, save_best_only=True, mode='max'),
    tfk.callbacks.TensorBoard(log_dir=f'{directory}', update_freq='epoch', histogram_freq=40, profile_batch='10,20')
]

print("Beginning training the best model...")

#best_model.load_weights(f'{directory}/weights/weights-final')

best_model.fit(train_iterator, epochs=501, callbacks=callbacks, validation_data=test_iterator)
best_model.save_weights(f'{directory}/weights/weights-final')
