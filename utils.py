# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.init as init
from mido import Message, MidiFile, MidiTrack

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


# Function for Initialization
def init_classic(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if m.__class__ in [nn.Conv1d, nn.ConvTranspose1d]:
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    if m.__class__ in [nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d]:
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif m.__class__ in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif m.__class__ in [nn.Linear]:
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif m.__class__ in [nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell]:
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)