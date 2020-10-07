from loader import load_noteseqs
from model import MVAE
import tensorflow.keras as tfk
from datetime import datetime
from shutil import copyfile
import os
import sys
import joblib
from generate_core import MGenerator
import signal
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import pypianoroll

def merge_dicts(*dicts):
    d = {}
    for dict in dicts:
        for key in dict:
            try:
                d[key].extend(dict[key])
            except KeyError:
                d[key] = dict[key]
    return d

def get_metrics_from_midi(path):
    metrics = {}

    try:
        track = pypianoroll.Multitrack(path)
        proll = track.get_merged_pianoroll()

        metrics["pitches"] = [pypianoroll.metrics.n_pitches_used(proll)]
        metrics["pitch_classes"] = [pypianoroll.metrics.n_pitch_classes_used(proll)]
        metrics["empty_beats"] = [pypianoroll.metrics.empty_beat_rate(proll, track.beat_resolution)]
        metrics["polyphony_1"] = [pypianoroll.metrics.polyphonic_rate(proll, threshold=1)]
        metrics["polyphony_2"] = [pypianoroll.metrics.polyphonic_rate(proll, threshold=2)]
        metrics["polyphony_3"] = [pypianoroll.metrics.polyphonic_rate(proll, threshold=3)]
        metrics["polyphony_4"] = [pypianoroll.metrics.polyphonic_rate(proll, threshold=4)]
    except:
        pass
    
    return metrics

def generate_and_save_samples(vae, epoch, path, n_genres):
    if epoch % 10 != 0:
        return

    save_path = "{}/samples/epoch-{}".format(path, epoch)

    os.makedirs(save_path, exist_ok=True)

    # TODO: Fix these magic numbers
    gen = MGenerator(21, 108, vae.x_depth, vae)

    genre_metrics = [{} for _ in range(n_genres)]

    for genre in range(n_genres):
        midis = gen.generate(genre, 10)
        metrics = {}

        for i, midi in enumerate(midis):
            midi.write("{}/genre-{}-{}.mid".format(save_path, genre, i))
            ms = get_metrics_from_midi("{}/genre-{}-{}.mid".format(save_path, genre, i))
            metrics = merge_dicts(metrics, ms)

        metrics = {
            "pitches_mean": np.average(metrics["pitches"]),
            "pitches_std": np.std(metrics["pitches"]),

            "pitch_classes_mean": np.average(metrics["pitch_classes"]),
            "pitch_classes_std": np.std(metrics["pitch_classes"]),

            "empty_beats_mean": np.average(metrics["empty_beats"]),
            "empty_beats_std": np.std(metrics["empty_beats"]),

            "polyphony_1_mean": np.average(metrics["polyphony_1"]),
            "polyphony_1_std": np.std(metrics["polyphony_1"]),

            "polyphony_2_mean": np.average(metrics["polyphony_2"]),
            "polyphony_2_std": np.std(metrics["polyphony_2"]),

            "polyphony_3_mean": np.average(metrics["polyphony_3"]),
            "polyphony_3_std": np.std(metrics["polyphony_3"]),

            "polyphony_4_mean": np.average(metrics["polyphony_4"]),
            "polyphony_4_std": np.std(metrics["polyphony_4"]),
        }
        genre_metrics[genre] = metrics

    series = []
    for i, stats in enumerate(genre_metrics):
        series.append(pd.Series(stats, name=str(i)))
    df_plot = pd.DataFrame(series)
    df_plot.to_csv("{}/stats.csv".format(save_path))

def create_model_and_train(train_segments, test_segments, x_depth, batch_size,
    enc_rnn_dim, dec_rnn_dim, enc_dropout, dec_dropout, cont_dim, cat_dim, mu_force, t_gumbel, style_embed_dim, kl_reg, beta_anneal_steps, rnn_type, attention,
    save_path, start_epoch, final_epoch, weights=None):

    train_iterator = load_noteseqs(train_segments, x_depth, batch_size).get_iterator()
    test_iterator = load_noteseqs(test_segments, x_depth, batch_size).get_iterator()

    vae = MVAE(x_depth=x_depth,
                enc_rnn_dim=enc_rnn_dim, enc_dropout=enc_dropout,
                dec_rnn_dim=dec_rnn_dim, dec_dropout=dec_dropout,
                cont_dim=cont_dim, cat_dim=cat_dim, mu_force=mu_force,
                t_gumbel=t_gumbel, style_embed_dim=style_embed_dim,
                kl_reg=kl_reg,
                beta_anneal_steps=beta_anneal_steps,
                rnn_type=rnn_type, attention=attention)

    optimizer = tfk.optimizers.Adam(learning_rate=5e-4)
    vae.compile(optimizer=optimizer)
    vae.run_eagerly = True

    now = datetime.now()
    save_path = save_path

    if(os.path.exists(save_path) == False):
        os.makedirs(save_path)

    callbacks = [
        tfk.callbacks.LambdaCallback(on_epoch_end=lambda epoch,_: generate_and_save_samples(vae, epoch, save_path, cat_dim)),
        tfk.callbacks.LambdaCallback(on_epoch_start=lambda epoch,_: vae.reset_trackers()),

        tfk.callbacks.EarlyStopping(monitor='val_p_acc', min_delta=0.01, patience=5, mode='max'),

        tfk.callbacks.CSVLogger(save_path + 'log.csv', append=True),    
        tfk.callbacks.ModelCheckpoint(save_path + 'weights/' + '/weights.{epoch:02d}', monitor='val_p_acc', save_weights_only=True, save_best_only=True, mode='max'),
        tfk.callbacks.TensorBoard(log_dir=save_path, write_graph=True, update_freq='epoch', histogram_freq=40, profile_batch='10,20')
    ]

    if weights != None:
        vae.load_weights(save_path + weights)

    history = vae.fit(train_iterator, epochs=final_epoch, initial_epoch=start_epoch, callbacks=callbacks, validation_data=test_iterator)
    vae.save_weights(save_path + 'weights/weights-final')

    return history
