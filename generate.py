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
from midi2audio import FluidSynth
from generate_core import MGenerator


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

vae = MVAE(x_depth=x_depth,
           enc_rnn_dim=args["enc_rnn_dim"], enc_dropout=args["enc_dropout"],
           dec_rnn_dim=args["dec_rnn_dim"], dec_dropout=args["dec_dropout"],
           cont_dim=args["cont_dim"], cat_dim=args["cat_dim"], mu_force=args["mu_force"],
           t_gumbel=args["t_gumbel"], style_embed_dim=args["style_embed_dim"],
           kl_reg=args["kl_reg"],
           beta_anneal_steps=args["kl_anneal"])


ap = argparse.ArgumentParser()
ap.add_argument("--restore_path", default=None, type=str)
ap.add_argument("--output_type", default="all", type=str)
ap.add_argument("--min_pitch", default=21, type=int)
ap.add_argument("--max_pitch", default=108, type=int)
ap.add_argument("--genre", default=0, type=int)
ap.add_argument("--n_generations", default=2, type=int)
ap.add_argument("--output_path", required=True, type=str)

args = ap.parse_args()
if args.restore_path:
    print('restoring weights from ' + args.restore_path)
    vae.load_weights(args.restore_path)


gen = MGenerator(args.min_pitch, args.max_pitch, x_depth, vae)
midis = gen.generate(args.genre, args.n_generations)

if(os.path.exists(args.output_path) == False):
    os.makedirs(args.output_path)

fs = FluidSynth()
for i, midi in enumerate(midis):
    if args.output_type == "midi":
        midi.write("{}/{}.mid".format(args.output_path, i))
    # elif args.output_type == "wav":
        # audio = midi.fluidsynth(44100, "piano.sf2").astype(np.float32)
        # fs.midi_to_audio(midi, args.output_path)
        # wavfile.write("{}/{}.wav".format(args.output_path, i), 44100, audio)
    elif args.output_type == "all":
        midi.write("{}/{}.mid".format(args.output_path, i))
        fs.midi_to_audio("{}/{}.mid".format(args.output_path, i), "{}/{}.wav".format(args.output_path, i))