import pretty_midi as pm
import tensorflow as tf
import numpy as np

class MGenerator:
    def __init__(self, min_pitch, max_pitch, x_depth, model):
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.END_TOKEN = max_pitch - min_pitch + 1
        self.x_depth = x_depth
        self.model = model

    def generate(self, genre, n_samples):
        logits = self.model.sample(genre, n_samples)

        midis = []
        for logit in logits:
            p, dt, d = self._logits_to_vecs(logit)
            midis.append(self._parse_vec_to_midi(p, dt, d))

        return midis

    def _logits_to_vecs(self, logits):
        p, dt, d = tf.split(logits, self.x_depth, axis=-1)
        p = tf.argmax(p, axis=-1)
        dt = tf.argmax(dt, axis=-1)
        d = tf.argmax(d, axis=-1)
        
        return p, dt, d

    def _parse_vec_to_midi(self, pitches, dts, durations, resolution=120, program=0, initial_tempo=100):
        
        # convert tensors to numpy if not already
        pitches = pitches.numpy() if isinstance(pitches, tf.Tensor) else pitches
        dts = dts.numpy() if isinstance(dts, tf.Tensor) else dts
        durations = durations.numpy() if isinstance(durations, tf.Tensor) else durations
        
        mid = pm.PrettyMIDI(resolution=resolution, initial_tempo=initial_tempo)
        instrument = pm.Instrument(program=program)
        
        start = 0

        for i in range(len(pitches)):
            pitch = int(pitches[i] + self.min_pitch)
            
            if int(pitches[i]) == self.END_TOKEN:
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