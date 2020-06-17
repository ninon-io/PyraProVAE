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

# Memory tracking if needed
# h = hpy()


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
        # number of tracks in data set
        self.nb_track = np.size(self.midi_files)
        # number of bars
        self.nb_bars = np.size(self.bar_files)

    def __len__(self):
        return self.nb_bars

    def __getitem__(self, index):
        return torch.load(self.bar_dir + '/' + self.bar_files[index])

    # Pre-processing of the data: loading in a sliced piano roll
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


# Main data import
def import_dataset(args):
    # Retrieve correct data loader
    if args.dataset == "maestro":  # Dataset is already splitted in 3 folders
        train_path = "/fast-1/mathieu/datasets/maestro_folders/train"
        test_path = "/fast-1/mathieu/datasets/maestro_folders/test"
        valid_path = "/fast-1/mathieu/datasets/maestro_folders/valid"
        train_set = PianoRollRep(train_path, args.frame_bar, export=False)
        test_set = PianoRollRep(test_path, args.frame_bar, export=False)
        valid_set = PianoRollRep(valid_path, args.frame_bar, export=False)
        train_indices, valid_indices = list(range(len(train_set))), list(range(len(valid_set)))
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

    elif args.dataset == "midi_folder":  # One folder with all midi files
        data_set = PianoRollRep(args.bar_dir, args.frame_bar, export=False)
        data_set_size = len(data_set)
        # compute indices for train/test split
        indices = np.array(list(range(data_set_size)))
        split = np.int(np.floor(args.test_size * data_set_size))
        if args.shuffle_data_set:
            np.random.seed(args.seed)
            np.random.shuffle(indices)
        global_train_indices, test_indices = np.array(indices[split:]), np.array(indices[:split])
        # Compute indices
        split = int(np.floor(args.valid_size * len(global_train_indices)))
        # Shuffle examples
        np.random.shuffle(global_train_indices)
        # Split the trainset to obtain a validation set
        train_indices, valid_indices = indices[split:], indices[:split]
        # create corresponding subsets
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
    else:
        print("Oh no, too bad: unknown dataset " + args.dataset + ".\n")
        exit()

    # Create all the loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.nbworkers,
                                               drop_last=True, sampler=train_sampler, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.nbworkers,
                                               drop_last=True, sampler=valid_sampler, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.nbworkers,
                                              drop_last=True, shuffle=False, pin_memory=True)
    return train_loader, valid_loader, test_loader, train_set, valid_set, test_set, args


