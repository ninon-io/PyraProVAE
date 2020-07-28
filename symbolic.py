# -*- coding: utf-8 -*-

import music21
from music21.stream import Voice
import music21.features.jSymbolic as jSymbolic
from mido import Message, MidiFile, MidiTrack

#%% ---------------------------------------------------------
#
# Symbolic features computation
#
# -----------------------------------------------------------

# Set of computable symbolic features
features = {
    'nb_notes':(None, 'int'),
    'min_duration':(jSymbolic.MinimumNoteDurationFeature, 'float'),
    'max_duration':(jSymbolic.MaximumNoteDurationFeature, 'float'),
    'note_density':(jSymbolic.NoteDensityFeature, 'float'),
    'average_duration':(jSymbolic.AverageNoteDurationFeature, 'float'),
    'quality':(jSymbolic.QualityFeature, 'binary'),
    'melodic_fifths':(jSymbolic.MelodicFifthsFeature, 'float'),
    'melodic_thirds':(jSymbolic.MelodicThirdsFeature, 'float'),
    'melodic_tritones':(jSymbolic.MelodicTritonesFeature, 'float'),
    'range':(jSymbolic.RangeFeature, 'int'),
    'average_interval':(jSymbolic.AverageMelodicIntervalFeature, 'float'),
    'average_attacks':(jSymbolic.AverageTimeBetweenAttacksFeature, 'float'),
    'pitch_variety':(jSymbolic.PitchVarietyFeature, 'int'),
    'amount_arpeggiation':(jSymbolic.AmountOfArpeggiationFeature, 'float'),
    'chromatic_motion':(jSymbolic.ChromaticMotionFeature, 'float'),
    'direction_motion':(jSymbolic.DirectionOfMotionFeature, 'float'),
    'melodic_arcs':(jSymbolic.DurationOfMelodicArcsFeature, 'float'),
    'melodic_span':(jSymbolic.SizeOfMelodicArcsFeature, 'float'),
    }

features_simple = {
    'nb_notes':(None, 'int'),
    'note_density':(jSymbolic.NoteDensityFeature, 'float'),
    'average_duration':(jSymbolic.AverageNoteDurationFeature, 'float'),
    'range':(jSymbolic.RangeFeature, 'int'),
    'average_interval':(jSymbolic.AverageMelodicIntervalFeature, 'float'),
    'pitch_variety':(jSymbolic.PitchVarietyFeature, 'int'),
    'amount_arpeggiation':(jSymbolic.AmountOfArpeggiationFeature, 'float'),
    'direction_motion':(jSymbolic.DirectionOfMotionFeature, 'float'),
    }

# Function to compute features from one given piano_roll
def symbolic_features(x_cur, feature_set=features, min_pitch=0):
    # First create a MIDI version
    midi = MidiFile(type = 1)
    midi.tracks.append(MidiTrack(roll_to_track(x_cur, min_pitch)))
    midi.save('/tmp/track.mid')
    # Then transform to a Music21 stream
    try:
        stream = music21.converter.parse('/tmp/track.mid')
    except: 
        feature_vals = {}
        feature_vals['nb_notes'] = 0
        for key, val in feature_set.items():
            feature_vals[key] = 0
        return feature_vals
    feature_vals = {}
    # Number of notes
    nb_notes = 0
    for n in stream.parts[0]:
        if (type(n) == Voice):
            continue
        if (n.isRest):
            continue
        nb_notes += 1
    feature_vals['nb_notes'] = nb_notes
    # Perform all desired jSymbolic extraction
    for key, val in feature_set.items():
        if (val[0] is None):
            continue
        try:
            feature_vals[key] = val[0](stream).extract().vector[0]
        except:
            feature_vals[key] = 0
    return feature_vals

# Function to compute all features from a given loader
def compute_symbolic_features(loader, args):
    final_features = {}
    for f in features:
        final_features[f] = []
    for x in loader:
        # Send to device
        x = x.to(args.device, non_blocking=True)
        for x_cur in x:
            # Compute symbolic features on input
            feats = symbolic_features(x_cur, min_pitch=args.min_pitch)
            for f in features:
                final_features[f].append(feats[f])
    return final_features

#%% ---------------------------------------------------------
#
# Symbolic export part
#
# -----------------------------------------------------------

def stop_note(note, time):
    return Message('note_off', note = note,
                   velocity = 0, time = time)

def start_note(note, time):
    return Message('note_on', note = note,
                   velocity = 127, time = time)

# Turn track into mido MidiFile
def roll_to_track(roll, midi_base=0):
    roll = roll.t()
    delta = 0
    # State of the notes in the roll.
    notes = [False] * len(roll[0])
    for row in roll:
        for i, col in enumerate(row):
            note = midi_base + i
            if col == 1:
                if notes[i]:
                    delta += 25
                    continue
                yield start_note(note, delta)
                delta = 0
                notes[i] = True
            elif col == 0:
                if notes[i]:
                    # Stop the ringing note
                    yield stop_note(note, delta)
                    delta = 0
                notes[i] = False
        if notes[i]:
            # Stop the ringing note
            yield stop_note(note, delta)
            delta = 0
        else:
            # ms per row
            delta += 25 