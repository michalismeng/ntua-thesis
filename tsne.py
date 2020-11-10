import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns
import json
from model import MVAE
import argparse
import tensorflow as tf
import joblib
from loader import load_noteseqs
import pretty_midi as pm

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

def parse_vec_to_midi(pitches, dts, durations, resolution=120, program=0, initial_tempo=100):
        
    # convert tensors to numpy if not already
    pitches = pitches.numpy() if isinstance(pitches, tf.Tensor) else pitches
    dts = dts.numpy() if isinstance(dts, tf.Tensor) else dts
    durations = durations.numpy() if isinstance(durations, tf.Tensor) else durations
    
    mid = pm.PrettyMIDI(resolution=resolution, initial_tempo=initial_tempo)
    instrument = pm.Instrument(program=program)
    
    start = 0

    for i in range(len(pitches)):
        pitch = int(pitches[i] + 21)
        
        if int(pitches[i]) == 88:
            break
        if i == 0:
            start = 0
        else:
            start += mid.tick_to_time(int(np.round((dts[i]/8)*mid.resolution)))
            
        velocity = 100
        duration = mid.tick_to_time(int(np.round((durations[i]/8)*mid.resolution)))
        end = start + duration
        
        note = pm.Note(pitch=pitch, velocity=velocity, start=start, end=end)
        instrument.notes.append(note)
    
    mid.instruments.append(instrument)    
    return mid

def logits_to_vecs(logits):
    p, dt, d = tf.split(logits, [89, 33, 33], axis=-1)
    p = tf.argmax(p, axis=-1)
    dt = tf.argmax(dt, axis=-1)
    d = tf.argmax(d, axis=-1)
        
    return p, dt, d

ap = argparse.ArgumentParser()
ap.add_argument("--restore_path", default=None, type=str)
ap.add_argument("--params_path", required=True, type=str)
ap.add_argument("--cat_dim", default=2, type=int)

x_depth = [89, 33, 33]

args = ap.parse_args()

with open(args.params_path, 'r') as f:
    params = "".join(f.readlines())

params = json.loads(params)

vae = MVAE(x_depth=x_depth,
        enc_rnn_dim=params['enc_rnn_dim'], enc_dropout=params['enc_dropout'],
        dec_rnn_dim=params['dec_rnn_dim'], dec_dropout=params['dec_dropout'],
        cont_dim=params['cont_dim'], cat_dim=args.cat_dim, mu_force=1.3,
        t_gumbel=params['t_gumbel'], style_embed_dim=params['style_dim'],
        kl_reg=params['kl_reg'],
        beta_anneal_steps=params['beta_anneal_steps'],
        rnn_type=params['rnn_type'], attention=params['attention'])

if args.restore_path:
    print('restoring weights from ' + args.restore_path)
    vae.load_weights(args.restore_path)

train_segments = []
test_segments = []
datasets = ["datasets/dataset-8/JSB-8-raw.pickle", "datasets/dataset-8/NMD-8-raw.pickle"]
keep_pcts = [1, 0.23]
master_pct = 1
for (dataset, pct) in zip(datasets, keep_pcts):
    segments = joblib.load(dataset)
    test_size = 0.1 * master_pct
    train_size = (1 - test_size) * master_pct
    Xtr, Xte = train_test_split(segments, train_size=train_size * pct, test_size=test_size * pct, random_state=42)
    train_segments.append(Xtr)
    test_segments.append(Xte)
    del segments

train_iterator = load_noteseqs(train_segments, x_depth, 600).get_iterator()
test_iterator = load_noteseqs(test_segments, x_depth, 600).get_iterator()

datas = []
labels = []
i = 0
for X, S, l in train_iterator:
    data = vae.embed_to_latent_mean(X, S)
    datas.append(data)
    labels.append(l)
    if i > 5:
        break
    i = i + 1

# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(datas[0])


# df = pd.DataFrame()
# df['tsne-2d-one'] = tsne_results[:, 0]
# df['tsne-2d-two'] = tsne_results[:, 1]
# df['label'] = ['JSB' if l == 0 else 'NMD' for l in labels[0]]

# df[df.label == 'JSB'].to_csv('tsne-JSB.csv')
# df[df.label == 'NMD'].to_csv('tsne-NMD.csv')

# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="label",
#     data=df,
#     legend="full",
#     alpha=0.3
# )

for X, S, l in train_iterator:
    Xa = X[:1,:,:]
    Sa = S[:1]
    for i in range(1, len(l)):
        if l[i] != l[0]:
            break 
    Xb = X[i:i+1,:,:]
    Sb = S[i:i+1]
    
print('labels:', l[0], l[i])

for a in [0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]:
    logits = vae.sample_interpolation(Xa, Xb, Sa, Sb, a)
    p, dt, d = logits_to_vecs(logits)

    midi = parse_vec_to_midi(p[0], dt[0], d[0])
    midi.write(f'test-{a}.mid')

# km = KMeans( n_clusters=2, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
# km.fit(tsne_results)

# print(km.cluster_centers_)

# plt.show()

