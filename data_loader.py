import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import math
import pretty_midi
from statistics import mean
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from guppy import hpy

data_dir = 'midi_short_dataset'
h = hpy()
batch_plot = 4
test_split = 0.2
shuffle_data_set = True
life_seed = 42


# Take the folder of midi files and output Piano-roll representation
class PianoRollRep(Dataset):
    def __init__(self, root_dir, frame_bar=100, export=False):
        # path directory with midi files
        self.root_dir = root_dir
        # files names .mid
        self.midi_files = np.array([files_names for files_names in os.listdir(root_dir)
                                    if (files_names.endswith('.midi') or files_names.endswith('.mid'))])
        # number of frame per bar
        self.frame_bar = frame_bar
        # path to the sliced piano-roll
        self.bar_dir = root_dir + "/piano_roll_bar_" + str(self.frame_bar)
        if not os.path.exists(self.bar_dir):
            os.mkdir(self.bar_dir)
            self.bar_export()
        else:
            if export:
                self.bar_export()
        # files names .pt
        self.bar_files = np.array([files_names for files_names in os.listdir(self.bar_dir)
                                   if files_names.endswith('.pt')])
        # number of tracks in dataset
        self.nb_track = np.size(self.midi_files)
        # number of bars
        self.nb_bars = np.size(self.bar_files)

    def __len__(self):
        return self.nb_bars

    def __getitem__(self, index):
        return torch.load(self.bar_dir + '/' + self.bar_files[index])

    def bar_export(self):
        # load midi in a pretty midi object
        for index in np.arange(start=0, stop=np.size(self.midi_files)):
            midi_data = pretty_midi.PrettyMIDI(self.root_dir + '/' + self.midi_files[index])
            downbeats = midi_data.get_downbeats()
            bar_time = mean([downbeats[i + 1] - downbeats[i] for i in range(len(downbeats) - 1)])
            fs = int(self.frame_bar / round(bar_time))
            piano_roll = midi_data.get_piano_roll(fs=fs)  # Returns np.array, shape=(128, times.shape[0])
            for i in range(len(downbeats) - 1):
                # compute the piano-roll for one bar and save it
                sliced_piano_roll = np.array(piano_roll[:,
                                             math.ceil(downbeats[i] * fs) + 1:math.ceil(downbeats[i + 1] * fs) + 1])
                if sliced_piano_roll.shape[1] > self.frame_bar:
                    sliced_piano_roll = np.array(sliced_piano_roll[:, 0:self.frame_bar])
                elif sliced_piano_roll.shape[1] < self.frame_bar:
                    sliced_piano_roll = np.pad(sliced_piano_roll,
                                               ((0, 0), (0, self.frame_bar - sliced_piano_roll.shape[1])), 'edge')
                sliced_piano_roll = torch.from_numpy(sliced_piano_roll).float()
                torch.save(sliced_piano_roll, self.bar_dir + "/per_bar" + str(i) + "_track" + str(index) + ".pt")


def get_data_loader(bar_dir, frame_bar=100, batch_size=16, export=False):
    data_set = PianoRollRep(bar_dir, frame_bar, export)
    data_set_size = len(data_set)
    # compute indices for train/test split
    indices = np.array(list(range(data_set_size)))
    split = np.int(np.floor(test_split * data_set_size))
    if shuffle_data_set:
        np.random.seed(life_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = np.array(indices[split:]), np.array(indices[:split])
    # create corresponding subsets
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=0, pin_memory=True, shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=test_sampler,
                                              num_workers=0, pin_memory=True, shuffle=False, drop_last=True)

    return train_loader, test_loader, train_sampler, test_sampler
