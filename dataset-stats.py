import pretty_midi as pm
import numpy as np
import joblib
import os
import glob
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

import pypianoroll

from dataset_stats_core import get_dataset_stats, get_dataset_metrics

bars_in_segment = 16
datasets = glob.glob(f"datasets/dataset-{bars_in_segment}/*-df.pickle")
series = []

def parse_dataset(dataset):
    df = joblib.load(dataset)
    name = dataset.split('/')[2].split('-df.pickle')[0]
    raw = joblib.load(f"datasets/dataset-{bars_in_segment}/{name}-raw.pickle")

    stats = get_dataset_stats(df)
    metrics = get_dataset_metrics(raw, name)
    stats.update(metrics)

    return (stats, name)


data = joblib.Parallel(n_jobs=-1, verbose=0)(joblib.delayed(parse_dataset)(dataset) for dataset in sorted(datasets))
for stats, name in data:
    series.append(pd.Series(stats, name=name))

df_plot = pd.DataFrame(series)
df_plot.to_csv(f"datasets/new-{bars_in_segment}-stats.csv")

print(df_plot)

# plt.figure()
# sns.barplot(df_plot.index, df_plot['song_count'])

# plt.figure()
# sns.barplot(df_plot.index, df_plot['segment_count'])

# plt.figure()
# sns.barplot(df_plot.index, df_plot['mean_events_per_segment'])


# plt.show()
