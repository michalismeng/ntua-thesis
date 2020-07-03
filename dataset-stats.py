import pretty_midi as pm
import numpy as np
import joblib
import os
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

jsb = joblib.load('dataset/JSB-df.pickle')
nmd = joblib.load('dataset/NMD-df.pickle')
jaz = joblib.load('dataset/JAZZ_MV-df.pickle')
mst = joblib.load('dataset/MAESTRO_18-df.pickle')

s1 = pd.Series(get_dataset_stats(jsb), name='JSB')
s2 = pd.Series(get_dataset_stats(nmd), name='NMD')
s3 = pd.Series(get_dataset_stats(jaz), name='JAZZ')
s4 = pd.Series(get_dataset_stats(mst), name='MAESTRO')

df_plot = pd.DataFrame([s1, s2, s3, s4])

plt.figure()
sns.barplot(df_plot.index, df_plot['song_count'])

plt.figure()
sns.barplot(df_plot.index, df_plot['segment_count'])

plt.figure()
sns.barplot(df_plot.index, df_plot['mean_events_per_segment'])


plt.show()
