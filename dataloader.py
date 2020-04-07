import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import math
import pretty_midi
from statistics import mean

data_dir = 'database/raw_midi'

batch_plot = 4
test_split = 0.2
shuffle_data_set = True
life_seed = 42


class PianoRollRep(Dataset):
    def __init__(self, midi_files, data_dir, frame_bar=100, transform=None):
        # files names .mid
        self.midi_files = midi_files
        # path directory with midi files
        self.data_dir = data_dir
        self.frame_bar = frame_bar
        # path to the sliced piano-roll
        self.bar_dir = data_dir + "/piano_roll_bar_" + str(self.frame_bar)
        self.transform = transform

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, index):
        return self.midi_frame[index]

    def bar_export(self):
        # load midi in a pretty midi object
        for index in range(len(self.midi_files)):
            midi_data = pretty_midi.PrettyMIDI(self.data_dir + '/' + self.midi_files[index])
            downbeats = midi_data.get_downbeats()
            size_downbeat = len(downbeats)-1
            for i in range(size_downbeat):
                bar_time = mean([downbeats[i+1] - downbeats[i]])
            fs = self.frame_bar / round(bar_time)
            piano_roll = midi_data.get_piano_roll(fs=fs)
            for i in range(size_downbeat):
                sliced_piano_roll = piano_roll[:, math.ceil(downbeats[i]*fs) + 1:math.ceil(downbeats[i+1]*fs) + 1]
                if sliced_piano_roll.shape[1] > self.frame_bar:
                    sliced_piano_roll = sliced_piano_roll[:, 0:self.frame_bar]
                elif sliced_piano_roll.shape[1] < self.frame_bar:
                    sliced_piano_roll = np.pad(sliced_piano_roll, ((0, 0), (0, self.frame_bar - sliced_piano_roll.shape[1])), 'edge')
                sliced_piano_roll = torch.Tensor(sliced_piano_roll)
                torch.save(sliced_piano_roll, self.bar_dir + "/per_bar" + str(i) + "_track" + str(index) + ".pt")







