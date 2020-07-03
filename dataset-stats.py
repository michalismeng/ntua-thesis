import pretty_midi as pm
import numpy as np
import joblib
import os
import glob
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

def get_dataset_stats(df):
    stats = {}
    stats['song_count'] = df.name.unique().size
    stats['segment_count'] = df.groupby(['name', 'segment_idx']).count().count().dataset
    stats['min_pitch'] = df.pitch.min()
    stats['max_pitch'] = df.pitch.max()
    stats['mean_events_per_segment'] = df[['name', 'segment_idx', 'pitch']].groupby(['name', 'segment_idx']).count().pitch.mean()
    stats['mean_song_length'] = df[['name', 'segment_idx']].groupby('name').count().segment_idx.mean()  # in segments
    
    return stats

datasets = glob.glob("dataset/*-df.pickle")
series = []
for dataset in sorted(datasets):
    df = joblib.load(dataset)
    name = dataset.split('/')[1].split('-df.pickle')[0]
    series.append(pd.Series(get_dataset_stats(df), name=name))

df_plot = pd.DataFrame(series)

print(df_plot)

plt.figure()
sns.barplot(df_plot.index, df_plot['song_count'])

plt.figure()
sns.barplot(df_plot.index, df_plot['segment_count'])

plt.figure()
sns.barplot(df_plot.index, df_plot['mean_events_per_segment'])


plt.show()
