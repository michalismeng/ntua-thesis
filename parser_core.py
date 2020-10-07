import pretty_midi as pm
import numpy as np
import math

from collections import Counter

### Configuration
# min/max pitch is determined by piano range
min_pitch = 21
max_pitch = 108
END_TOKEN = max_pitch - min_pitch + 1
################

# One hot encoder from paper
class OneHotEncoder:
    def __init__(self, depth, axis=-1):
        self.depth = depth
        self.axis = axis
        
    def _onehot(self, data):
        oh = np.zeros((self.depth), dtype=np.uint8)
        if data >= 0 and data < self.depth:
            data = int(data)
            oh[data] = 1
        else:
            print('onehot error')
        return oh
    
    def transform(self, data_list):
        one_hot_encoded = [self._onehot(data) for data in data_list]
        one_hot_encoded = np.stack(one_hot_encoded, axis=0)
        
        return one_hot_encoded

# split a onehot vector to the three features we care about
def onehot_vec_to_features(vec):
    pitch_length = max_pitch - min_pitch + 1 + 1
    ps, dts, durs = [], [], [] 
    for x in vec:
        x1 = np.split(x, [pitch_length, pitch_length + 33], axis=0)
        ps.append(np.argmax(x1[0]))
        dts.append(np.argmax(x1[1]))
        durs.append(np.argmax(x1[2]))

    return ps, dts, durs

# parse the output of the neural network back to a midi representation
def parse_vec_to_midi(pitches, dts, durations, resolution=120, program=0, initial_tempo=100):
    mid = pm.PrettyMIDI(resolution=resolution, initial_tempo=initial_tempo)
    instrument = pm.Instrument(program=program)
    
    start = 0
    for i in range(len(pitches)):
        if int(pitches[i]) == END_TOKEN:
            break
        if i == 0:
            start = 0
        else:
            start += mid.tick_to_time(int(np.round((dts[i]/8)*mid.resolution)))
            
        pitch = int(pitches[i] + min_pitch)
        velocity = 100
        duration = mid.tick_to_time(int(np.round((durations[i]/8)*mid.resolution)))
        end = start + duration
        
        note = pm.Note(pitch=pitch, velocity=velocity, start=start, end=end)
        instrument.notes.append(note)
    
    mid.instruments.append(instrument)    
    return mid

# parse a midi file to a suitable representation for the neural network
def parse_midi_file(song_path, bars_per_segment=4):
    try:
        mid = pm.PrettyMIDI(song_path)
    except:
        print('Could not parse', song_path)
        return None
    
    numerators = [t.numerator for t in mid.time_signature_changes]
    denominators = [t.denominator for t in mid.time_signature_changes]
    countN = Counter(numerators)
    countD = Counter(denominators)
    
    if len(countN) and len(countD):
        numerator = sorted(numerators, key=lambda x: countN[x], reverse=True)[0]
        denominator = sorted(denominators, key=lambda x: countD[x], reverse=True)[0]  
    else:
        numerator = 4
        denominator = 4
        
    # extract all notes from non-drum instruments
    midi_notes = []
    for ins in mid.instruments:
        if not ins.is_drum:
            for n in ins.notes:
                midi_notes.append((n.pitch, n.start, n.end))

    if len(midi_notes) == 0:
        return None
                
    # sort by (start time, pitch)
    midi_notes = sorted(midi_notes, key=lambda x: (x[1], x[0]))
    
    # (pitch, dt, duration, start time)  -- all times in beats
    song = [(midi_notes[0][0], mid.time_to_tick(midi_notes[0][1])/mid.resolution,
             mid.time_to_tick(midi_notes[0][2] - midi_notes[0][1])/mid.resolution, midi_notes[0][1])]
    
    for m, m_prev in zip(midi_notes[1:], midi_notes[:-1]):
        t = mid.time_to_tick(m[1]) - mid.time_to_tick(m_prev[1])
        p = m[0]
        song.append((p, t/mid.resolution, mid.time_to_tick(m[2] - m[1])/mid.resolution, m[1]))
        
    # create 4-bar segments
    time_per_segment = mid.tick_to_time((bars_per_segment * numerator * mid.resolution) // denominator) # time per segment
    total_bars = int((mid.get_end_time() // time_per_segment))
    
    segments = [[] for _ in range(total_bars)]
    for m in song:
        i = int(m[-1] // time_per_segment // 4)
        segments[i].append([m[0], m[1], m[2]])

    def validate_segment(s):
        if len(s) < 5:
            return False
        if any([p < min_pitch or p > max_pitch for p,_,_ in s]):
            return False
        return True

    # keep only segments that have more than 5 note events
    segments = [np.stack(b) for b in segments if validate_segment(b)]
    
    # one hot encoders that round floats as required
    p_ohe = OneHotEncoder(max_pitch - min_pitch + 1 + 1)   # add one for the END_TOKEN mark  (and remember that clipping pitches is inclusive like tomasulo)
    t_ohe = OneHotEncoder(33)
    
    # holds a list of segments
    X = []
    for segment in segments:
        # each segment holds 3-piece features. Split them into 3 arrays and flatten
        features_split = np.split(segment, 3, -1)
        features_split = [np.squeeze(x) for x in features_split]
        # features_split has 3 arrays, namely Pitch, Dt, Duration for the 4-bar segment
        
        P, Dt, D = [], [], []
        
        p = p_ohe.transform(features_split[0] - min_pitch)
        
        dt = np.minimum(np.round((features_split[1]/4) * 32), 32)   # divide by four since a whole note has four beats
        dt = t_ohe.transform(dt)

        #TODO: a two beat note when rounded results in something less than a two-beat note (because it starts as less than)
        d = np.minimum(np.round((features_split[2]/4) * 32), 32)
        d = t_ohe.transform(d)

        P.append(p)
        Dt.append(dt)
        D.append(d)
        
        P = np.concatenate(P, axis=0)
        Dt = np.concatenate(Dt, axis=0)
        D = np.concatenate(D, axis=0)
    
        tmp = np.concatenate([P, Dt, D], -1)
        end_token = np.zeros(dtype=np.float32, shape=(1, tmp.shape[-1]))
        end_token[0, END_TOKEN] = 1.0   # index into a 2D array
        tmp = np.concatenate([tmp, end_token], 0)
        tmp = tmp.astype(np.uint8)
            
        X.append(tmp)
        
    return segments, X