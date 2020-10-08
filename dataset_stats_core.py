import pypianoroll
import tensorflow as tf
import numpy as np
import pretty_midi as pm

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

# returns dataset statistics from dataframe structure
def get_dataset_stats(df):
    stats = {}
    stats['song_count'] = df.name.unique().size
    stats['segment_count'] = df.groupby(['name', 'segment_idx']).count().count().dataset
    stats['min_pitch'] = df.pitch.min()
    stats['max_pitch'] = df.pitch.max()
    stats['mean_events_per_segment'] = df[['name', 'segment_idx', 'pitch']].groupby(['name', 'segment_idx']).count().pitch.mean()
    stats['std_events_per_segment'] = df[['name', 'segment_idx', 'pitch']].groupby(['name', 'segment_idx']).count().pitch.std()
    stats['mean_song_length'] = df[['name', 'segment_idx']].groupby('name').count().segment_idx.mean()  # in segments
    stats['std_song_length'] = df[['name', 'segment_idx']].groupby('name').count().segment_idx.std()  
    
    return stats

# returns metrics from a midi file
def get_metrics_from_midi(path):
    metrics = {}

    try:
        track = pypianoroll.Multitrack(path)
        proll = track.get_merged_pianoroll()

        metrics["pitches"] = [pypianoroll.metrics.n_pitches_used(proll)]
        metrics["pitch_classes"] = [pypianoroll.metrics.n_pitch_classes_used(proll)]
        metrics["empty_beats"] = [pypianoroll.metrics.empty_beat_rate(proll, track.beat_resolution)]
        metrics["polyphony_1"] = [pypianoroll.metrics.polyphonic_rate(proll, threshold=1)]
        metrics["polyphony_2"] = [pypianoroll.metrics.polyphonic_rate(proll, threshold=2)]
        metrics["polyphony_3"] = [pypianoroll.metrics.polyphonic_rate(proll, threshold=3)]
        metrics["polyphony_4"] = [pypianoroll.metrics.polyphonic_rate(proll, threshold=4)]
    except:
        pass
    
    return metrics

# merges dictionaries
def merge_dicts(*dicts):
    d = {}
    for dict in dicts:
        for key in dict:
            try:
                d[key].extend(dict[key])
            except KeyError:
                d[key] = dict[key]
    return d

# returns dictionary of mean and std of the given metrics
def get_average_metrics(metrics):
    return {
        "pitches_mean": np.average(metrics["pitches"]),
        "pitches_std": np.std(metrics["pitches"]),

        "pitch_classes_mean": np.average(metrics["pitch_classes"]),
        "pitch_classes_std": np.std(metrics["pitch_classes"]),

        "empty_beats_mean": np.average(metrics["empty_beats"]),
        "empty_beats_std": np.std(metrics["empty_beats"]),

        "polyphony_1_mean": np.average(metrics["polyphony_1"]),
        "polyphony_1_std": np.std(metrics["polyphony_1"]),

        "polyphony_2_mean": np.average(metrics["polyphony_2"]),
        "polyphony_2_std": np.std(metrics["polyphony_2"]),

        "polyphony_3_mean": np.average(metrics["polyphony_3"]),
        "polyphony_3_std": np.std(metrics["polyphony_3"]),

        "polyphony_4_mean": np.average(metrics["polyphony_4"]),
        "polyphony_4_std": np.std(metrics["polyphony_4"]),
    }
    
# returns dataset metrics from raw structure
def get_dataset_metrics(raw, name):
    metrics = {}
    for logits in raw:
        midi = parse_vec_to_midi(*logits_to_vecs(logits))
        midi.write(f"temp/{name}.mid")

        temp = get_metrics_from_midi(f"temp/{name}.mid")
        metrics = merge_dicts(metrics, temp)

    print(f"total segments for {name}: ", len(raw), "parsed metrics: ", len(metrics["pitches"]))

    return get_average_metrics(metrics)