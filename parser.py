import joblib
import glob
import os
import pandas as pd
import numpy as np
import pretty_midi as pm
import argparse
from parser_core import onehot_vec_to_features, parse_midi_file, parse_vec_to_midi, max_pitch, min_pitch

from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", type=str, required=True)
ap.add_argument("--dataset_name", type=str, required=True)
ap.add_argument("--save_path", type=str, required=True)
ap.add_argument("--bars_per_segment", type=str, required=True)

args = ap.parse_args()

datasets = args.dataset.split()
names = args.dataset_name.split()
bars_per_segment = [int(b) for b in args.bars_per_segment.split()]

flatten = lambda l: [item for sublist in l for item in sublist]

# returns a dataframe of the segments produces and a raw array suitable for input to the neural network
def build_dataframe_for_dataset(dataset_name, files, save_path, bars_per_segment, load_pickle=False, save_pickle=False):
    def parse_and_combine_name(midi_file):
        return (midi_file.split('/')[-1], parse_midi_file(midi_file, bars_per_segment))
    
    if load_pickle:
        data_df = joblib.load(save_path + '{}-df.pickle'.format(dataset_name))
        data_raw = joblib.load(save_path + '{}-raw.pickle'.format(dataset_name))
    else:
        data = joblib.Parallel(n_jobs=-1, verbose=0)(
            joblib.delayed(parse_and_combine_name)(midi_file) for midi_file in files)
        data = [(n, d) for n, d in data if d is not None]
        
        data_df = [(n, s) for (n, (s, _)) in data]
        data_raw = [X for (n, (_, X)) in data]
        data_raw = flatten(data_raw)
        
    formatted = []
    for song_name, song in data_df:
        for s_idx, segment in enumerate(song):
            for feature in segment:
                formatted.append((dataset_name, song_name, s_idx, feature[0], feature[1], feature[2]))
    data_df = pd.DataFrame(formatted, columns=['dataset', 'name', 'segment_idx', 'pitch', 'dt', 'duration'])

    os.makedirs(save_path, exist_ok=True)

    if save_pickle:
        joblib.dump(data_df, save_path + '{}-df.pickle'.format(dataset_name))
        joblib.dump(data_raw, save_path + '{}-raw.pickle'.format(dataset_name))

    return data_df, data_raw

for dataset, name, bps in zip(datasets, names, bars_per_segment):
    files = glob.glob(dataset + '/*')
    df, raw = build_dataframe_for_dataset(name, files, args.save_path, bps, save_pickle=True)