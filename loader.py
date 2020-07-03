import joblib
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences 

class load_noteseqs:
    def __init__(self, list_of_segments, x_depth, batch_size=16):
        # self.data = [joblib.load(p) for p in paths]
        
        self.notes = [d for d in list_of_segments]    
        
        self.labels = []
        for i in range(len(list_of_segments)):  # each path is a different music genre
            num_segments = len(self.notes[i])
            tmp_labels = np.ones(shape=[num_segments]) * i
            self.labels.append(tmp_labels)

        self.labels = self.labels[0] if len(self.labels) == 1 else np.concatenate(self.labels, 0)
        
        self.x_depth = x_depth
        self.notes = list(itertools.chain.from_iterable(self.notes))
    
        self.seq_len = [len(x) for x in self.notes]
        
        self.batch_size = int(batch_size)
        self.total_batches = int(len(self.notes) // self.batch_size)
        
        self.generator = tf.random.Generator.from_seed(42)

    def loader(self):
        Z = list(zip(self.notes, self.seq_len, self.labels))
        np.random.shuffle(Z)    # shuffle segments
        notes, seq_len, labels = zip(*Z)

        for i in range(self.total_batches):
            tmp_notes = notes[self.batch_size*i:(self.batch_size*i)+self.batch_size]  # one batch of notes
            tmp_seq_len = seq_len[self.batch_size*i:(self.batch_size*i)+self.batch_size] # one batch of sequence lengths
            tmp_labels = labels[self.batch_size*i:(self.batch_size*i)+self.batch_size] # one batch of sequence lengths
            if len(tmp_notes) < self.batch_size:   # we are done
                break
            else:
                tmp_notes = pad_sequences(tmp_notes, padding="post", dtype=np.int32, value=-1)
                yield tmp_notes, tmp_seq_len, tmp_labels
                
    def augment(self, batch_n, batch_s, batch_l):
        aug = self.generator.uniform(shape=(), minval=-5.9, maxval=6)
        aug = tf.cast(aug, tf.int32)
        
        pitch = tf.roll(batch_n[:, :, :88], aug, axis=-1)
        aug_notes = tf.concat([pitch, batch_n[:, :, 88:]], -1)
        
        return aug_notes, batch_s, batch_l
                
    def get_iterator(self):
        ds = tf.data.Dataset.from_generator(self.loader, (tf.float32, tf.int32, tf.int32))
        ds = ds.cache()
        ds = ds.shuffle(self.batch_size*2)                # shuffle batches
        ds = ds.map(self.augment, num_parallel_calls=2)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        
        return ds

if __name__ == "__main__":
    x = joblib.load("dataset/JSB-raw.pickle")
    y = load_noteseqs(x, [89, 33, 33], batch_size=64).loader()

    ds_notes = tf.data.Dataset.from_tensor_slices(y[0])
    ds_len = tf.data.Dataset.from_tensor_slices(y[1])
    ds_labels = tf.data.Dataset.from_tensor_slices(y[2])

    ds = tf.data.Dataset.zip(ds_notes, ds_len, ds_labels)