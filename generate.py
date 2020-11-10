from loader import load_noteseqs
from model import MVAE
import argparse
import tensorflow.keras as tfk
from datetime import datetime
import shlex
from shutil import copyfile
import os
import sys
import numpy as np
from scipy.io import wavfile
from generate_core import MGenerator
import json

ap = argparse.ArgumentParser()
ap.add_argument("--restore_path", default=None, type=str)
ap.add_argument("--params_path", required=True, type=str)
ap.add_argument("--genre", default=0, type=int)
ap.add_argument("--cat_dim", default=2, type=int)
ap.add_argument("--n_generations", default=1, type=int)
ap.add_argument("--output_path", default="test-gen", type=str)

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


gen = MGenerator(21, 108, x_depth, vae)
midis = gen.generate(args.genre, args.n_generations)

if(os.path.exists(args.output_path) == False):
    os.makedirs(args.output_path)

for i, midi in enumerate(midis):
    midi.write("{}/{}.mid".format(args.output_path, i))