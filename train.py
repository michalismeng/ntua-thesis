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

def generate_and_save_samples(vae, epoch, path, n_genres):
    if epoch % 40 != 0:
        return

    save_path = "{}/samples/epoch-{}".format(path, epoch)

    os.makedirs(save_path, exist_ok=True)

    # TODO: Fix these magic numbers
    gen = MGenerator(21, 108, vae.x_depth, vae)
    for genre in range(n_genres):
        midis = gen.generate(genre, 5)
        for i, midi in enumerate(midis):
            midi.write("{}/genre-{}-{}.mid".format(save_path, genre, i))
    

def parse_configuration(config):
    args = {}
    with open(config, 'r') as f:
        lines = list(line for line in (l.strip() for l in f) if line)
    for line in lines:
        k, v = shlex.split(line)  # shlex to ignore in-quote spaces
        args[k] = v
    return args

# capture Ctrl+C
def signal_handler(sig, frame):
    print('saving final weights...')
    vae.save_weights(save_path + 'weights/weights-final')
    sys.exit(0)

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
    name = dataset.split('/')[1].split('-raw.pickle')[0] 
    print(name, len(train_segment), len(test_segment))

input("Press Enter to continue...")

if int(args["cat_dim"]) != len(train_segments):
    print('{} = cat_dim != number of different datasets = {}'.format(args["cat_dim"], len(train_segments)))
    exit(1)

# setup train/test dataset
train_iterator = load_noteseqs(train_segments, x_depth, batch_size=args["batch_size"]).get_iterator()
test_iterator = load_noteseqs(test_segments, x_depth, batch_size=args["batch_size"]).get_iterator()

vae = MVAE(x_depth=x_depth,
           enc_rnn_dim=args["enc_rnn_dim"], enc_dropout=args["enc_dropout"],
           dec_rnn_dim=args["dec_rnn_dim"], dec_dropout=args["dec_dropout"],
           cont_dim=args["cont_dim"], cat_dim=args["cat_dim"], mu_force=args["mu_force"],
           t_gumbel=args["t_gumbel"], style_embed_dim=args["style_embed_dim"],
           kl_reg=args["kl_reg"],
           beta_anneal_steps=args["kl_anneal"],
           rnn_type=rnn_type, attention=attention)

optimizer = tfk.optimizers.Adam(learning_rate=5e-4)
vae.compile(optimizer=optimizer)
vae.run_eagerly = True

now = datetime.now()
save_path = args["save_path"]

# copy configuration file in model directory
if(os.path.exists(save_path) == False):
    os.makedirs(save_path)
copyfile(config_file, save_path + 'train.conf') 

with open(save_path + 'model.txt', 'w') as f:
    vae.model().summary(print_fn=lambda x: f.write(x + '\n'))

# register handler for Ctrl+C in order to save final weights
signal.signal(signal.SIGINT, signal_handler)

# print('loading weights')
# vae.load_weights('gru_jsb_nmd/weights/weights-final')

callbacks = [
    tfk.callbacks.LambdaCallback(on_epoch_end=lambda epoch,_: generate_and_save_samples(vae, epoch, save_path, int(args["cat_dim"]))),
    tfk.callbacks.LambdaCallback(on_epoch_start=lambda epoch,_: vae.reset_trackers()),

    tfk.callbacks.CSVLogger(save_path + 'log.csv', append=True),    
    tfk.callbacks.ModelCheckpoint(save_path + 'weights/' + '/weights.{epoch:02d}', monitor='val_p_acc', save_weights_only=True, save_best_only=True, mode='max'),
    tfk.callbacks.TensorBoard(log_dir=save_path, write_graph=True, update_freq='epoch', histogram_freq=40, profile_batch='10,20')
]

history = vae.fit(train_iterator, epochs=int(args["epochs"]), callbacks=callbacks, validation_data=test_iterator)
# history = vae.fit(train_iterator, epochs=1001, initial_epoch=800, callbacks=callbacks, validation_data=test_iterator)

vae.save_weights(save_path + 'weights/weights-final')