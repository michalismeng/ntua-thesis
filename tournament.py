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

import pypianoroll
import itertools
import random

from tournament_core import create_model_and_train

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


enc_dims = [64, 128, 256, 512]
dropouts = [0.3, 0.4, 0.5, 0.6, 0.7]
cont_dims = [50, 120, 170, 250]
mu_forces = [0.5, 1.2, 1.8, 5]
t_gumbels = [0.001, 0.02, 0.1]
style_dimss = [20, 80, 150]
kl_regs = [0.2, 0.8, 1.5]
beta_steps = [500, 1000, 1800, 3000]
attentions = [0, 128, 256, 512]

max_combination = 3
max_turns = 2
epochs_per_turn = 5

combinations = list(itertools.product(enc_dims, dropouts, cont_dims, mu_forces, t_gumbels, style_dimss, kl_regs, beta_steps, attentions))
selected_combinations = [random.choice(combinations) for _ in range(max_combination)]
results = [0 for _ in range(len(selected_combinations))]
banned = [False for _ in range(len(selected_combinations))]

# for turn in range(10):
for turn in range(max_turns):
    for i, comb in enumerate(selected_combinations):
        enc_dim, dropout, cont_dim, mu_force, t_gumbel, style_dim, kl_reg, beta_step, attention = comb

        if banned[i]:
            continue

        print("**************************************************************************")
        print("**************************************************************************")
        print("Running configuration:", str(comb))
        print("**************************************************************************")
        print("**************************************************************************")

        save_path = f'tournament/{i}/'

        if(os.path.exists(save_path) == False):
            os.makedirs(save_path)
        
        with open(save_path + 'train_params', 'w') as f:
            f.write(str(comb))

        history = create_model_and_train(train_segments, test_segments, x_depth, 64, enc_dim, enc_dim, dropout, dropout, cont_dim, 2, mu_force, t_gumbel, style_dim, kl_reg,
            beta_step, 'lstm', attention, save_path, epochs_per_turn * turn, epochs_per_turn * turn + epochs_per_turn)
        results[i] = history.history['val_p_acc']

    min_elem = np.argmin(results)
    results[min_elem] = np.inf
    banned[min_elem] = True

print("Final results")
print(results)
print(np.argmin(results))