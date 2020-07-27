# %%

import torch
from torch.utils.data.dataset import Dataset
from torch import nn
import os
import numpy as np
import math
import pretty_midi
from statistics import mean
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import torchvision.transforms as transform
from torchvision.transforms import functional
from .transforms import Transpose, MaskColumns, MaskRows, PitchFlip, TimeFlip
#from guppy import hpy
import argparse
from idlelib.pyparse import trans
import random
import matplotlib.pyplot as plt


def maximum(train_set, valid_set, test_set):
    # Compute the maximum of the dataset
    max_v = 0
    for s in [train_set, valid_set, test_set]:
        for x in s:
            max_v = torch.max(torch.tensor([torch.max(x), max_v]))
    max_global = max_v
    track_train = []
    track_valid = []
    track_test = []
    for x in train_set:
        x_norm = torch.div(x, max_global)
        track_train.append(x_norm)
    for y in valid_set:
        y_norm = torch.div(y, max_global)
        track_valid.append(y_norm)
    for z in test_set:
        z_norm = torch.div(z, max_global)
        track_test.append(z_norm)
    return max_global, track_train, track_valid, track_test


# Main data import
def import_dataset(args):
    base_path = args.midi_path
    # Main transform
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=2.342, std=12.476)])  # Rescale?
    folder_str = {'maestro': 'maestro_folders', 'nottingham': 'Nottingham', 'bach': 'JSB_Chorales'}
    base_path += '/' + folder_str[args.dataset]
    # Retrieve correct data loader
    if args.dataset in ["maestro", "nottingham", "bach_chorales"]:
        train_path = base_path + "/train"
        test_path = base_path + "/test"
        valid_path = base_path + "/valid"
        # Import each of the set
        train_set = PianoRollRep(train_path, args.frame_bar, args.score_type, args.score_sig, args.data_binarize,
                                 args.data_augment, args.data_export)
        test_set = PianoRollRep(test_path, args.frame_bar, args.score_type, args.score_sig, args.data_binarize,
                                args.data_augment, args.data_export, False)
        valid_set = PianoRollRep(valid_path, args.frame_bar, args.score_type, args.score_sig, args.data_binarize,
                                 args.data_augment, args.data_export, False)
        # Normalization
        if args.data_normalize:
            min_v, max_v, min_p, max_p, vals = stats_dataset([train_set, valid_set, test_set])
            for sampler in [train_set, valid_set, test_set]:
                sampler.max_v = max_v
                if args.data_pitch:
                    sampler.min_p = min_p
                    sampler.max_p = max_p
        # Get sampler
        train_indices, valid_indices, test_indices = list(range(len(train_set))), list(range(len(valid_set))), \
                                                     list(range(len(test_set)))
        if args.subsample > 0:
            train_indices = train_indices[:args.subsample]
            valid_indices = valid_indices[:args.subsample]
            test_indices = test_indices[:args.subsample]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        test_sampler = SubsetRandomSampler(test_indices)
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
                                              drop_last=True, sampler=test_sampler, shuffle=False, pin_memory=True)
    batch = next(iter(train_loader))
    args.input_size = batch[0].shape
    return train_loader, valid_loader, test_loader, train_set, valid_set, test_set, args


# Take the folder of midi files and output Piano-roll representation
class PianoRollRep(Dataset):
    def __init__(self, root_dir, frame_bar=64, score_type='all', score_sig='all', binarize=False, augment=False,
                 export=False, training=True):
        # path directory with midi files
        self.root_dir = root_dir
        # files names .mid
        self.midi_files = np.array([files_names for files_names in os.listdir(root_dir) if
                                    (files_names.endswith('.midi') or files_names.endswith('.mid'))])
        # number of frame per bar
        self.frame_bar = frame_bar
        # Type of score (mono or all)
        self.score_type = score_type
        # Time signature of the score
        self.score_sig = score_sig
        # Binarize the data or not
        self.binarize = binarize
        # Check if this is a train set
        self.training = training
        # Data augmentation
        self.augment = augment
        self.transform = transform.RandomApply(
            [transform.RandomChoice([Transpose(6), MaskRows(), TimeFlip(), PitchFlip()])], p=.3)
        # Base values for eventual normalization
        self.min_p = 0
        self.max_p = 128
        self.max_v = 1.
        # path to the sliced piano-roll
        self.bar_dir = root_dir + "/piano_roll_bar_" + str(
            self.frame_bar) + '_' + self.score_type + '_' + self.score_sig
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
        cur_track = torch.load(self.bar_dir + '/' + self.bar_files[index])
        cur_track /= self.max_v
        if self.binarize:
            cur_track[cur_track > 0] = 1
        output = cur_track[self.min_p:(self.max_p + 1), :]
        if self.augment and self.training:
            output = self.transform(output)
        return output

    # Find the perfect match transforms, transforms.functional.center_crop((0, 0), (128, 100, 0))
    # def __getitem__(self, index):
    #     transform = transforms.Compose([transforms.Normalize([0.5], [0.5])])
    #     sample = torch.load(self.bar_dir + '/' + self.bar_files[index])
    #     norm_sample = transform(sample.unsqueeze(0))
    #     return norm_sample

    # Pre-processing of the data: loading in a sliced piano roll
    def bar_export(self):
        if self.score_sig != 'all':
            sig_split = self.score_sig.split('_')
            target_sig_n = int(sig_split[0])
            target_sig_d = int(sig_split[1])
        # load midi in a pretty midi object
        for index in np.arange(start=0, stop=np.size(self.midi_files)):
            midi_data = pretty_midi.PrettyMIDI(self.root_dir + '/' + self.midi_files[index])
            ts_n = midi_data.time_signature_changes[0].numerator
            ts_d = midi_data.time_signature_changes[0].denominator
            # Eventually check for time signature
            if self.score_sig != 'all' and (ts_n != target_sig_n or ts_d != target_sig_d):
                print('Signature is [%d/%d] - skipped as not a 4/4 track' % (ts_n, ts_d))
                continue
            downbeats = midi_data.get_downbeats()
            bar_time = mean([downbeats[i + 1] - downbeats[i] for i in range(len(downbeats) - 1)])
            fs = int(self.frame_bar / round(bar_time))
            # Find a mono track if we only want a mono dataset
            if self.score_type == 'mono':
                found_track = 0
                for i in range(len(midi_data.instruments)):
                    piano_roll = midi_data.instruments[i].get_piano_roll(fs=fs)
                    piano_roll_bin = piano_roll.copy()
                    piano_roll_bin[piano_roll_bin > 0] = 1
                    if np.sum(np.sum(piano_roll_bin, axis=0) > 1) == 0:
                        found_track = 1
                        break
                if found_track == 0:
                    continue
            else:
                # Otherwise take all tracks at once
                piano_roll = midi_data.get_piano_roll(fs=fs)
            for i in range(len(downbeats) - 1):
                # compute the piano-roll for one bar and save it
                sliced_piano_roll = np.array(piano_roll[:,
                                             math.ceil(downbeats[i] * fs):math.ceil(downbeats[i + 1] * fs)])
                if sliced_piano_roll.shape[1] > self.frame_bar:
                    sliced_piano_roll = np.array(sliced_piano_roll[:, 0:self.frame_bar])
                elif sliced_piano_roll.shape[1] < self.frame_bar:
                    # sliced_piano_roll = np.pad(sliced_piano_roll, ((0, 0), (0, self.frame_bar - sliced_piano_roll.shape[1])), 'edge')
                    continue
                sliced_piano_roll = torch.from_numpy(sliced_piano_roll).float()
                torch.save(sliced_piano_roll, self.bar_dir + "/per_bar" + str(i) + "_track" + str(index) + ".pt")


def test_data(args, batch):
    # Plot settings
    nrows, ncols = 2, 2  # array of sub-plots
    figsize = np.array([8, 20])  # figure size, inches
    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i, axi in enumerate(ax.flat):
        piano_roll = batch[i]
        axi.matshow(piano_roll, alpha=1)
    fig.show()


def stats_dataset(loaders):
    max_v, min_v, val, pitch_on, count_mono, count_poly = 0, 3000, {}, [], 0, 0
    for cur_loader in loaders:
        train_val = cur_loader.training
        cur_loader.training = False
        for x in cur_loader:
            max_v = max((torch.max(x), max_v))
            min_v = min((torch.min(x), min_v))
            val_t, counts = torch.unique(x, return_counts=True)
            for i, v in enumerate(val_t):
                v_c = int(v.item())
                if val.get(v_c) is None:
                    val[v_c] = 0
                val[v_c] += counts[i]
            x_sum = torch.sum(x, dim=1)
            pitch_on.append(torch.nonzero(x_sum))
            x[x > 0] = 1
            if torch.sum(torch.sum(x, dim=0) > 1):
                count_poly += 1
            else:
                count_mono += 1
        cur_loader.training = train_val
    min_p, max_p = int(min(pitch_on)), int(max(pitch_on))
    if (min_p > 5):
        min_p -= 6
    if (max_p < 122):
        max_p += 6
    pitch_on = torch.unique(torch.cat(pitch_on))
    print('*' * 32)
    print('Dataset summary')
    print('Min : %d' % min_v)
    print('Max : %d' % max_v)
    print('Velocity values')
    print(val)
    print('Pitch ons')
    print(pitch_on)
    print('Poly : %d' % count_poly)
    print('Mono : %d' % count_mono)
    return float(min_v), float(max_v), min_p, max_p, val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataloader')
    # Data Parameters
    parser.add_argument('--midi_path', type=str, default='/Users/esling/Datasets/symbolic/', help='path to midi folder')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--subsample', type=int, default=0, help='train on subset')
    parser.add_argument("--test_size", type=float, default=0.2, help="% of data used in test set")
    parser.add_argument("--valid_size", type=float, default=0.2, help="% of data used in valid set")
    parser.add_argument("--dataset", type=str, default="nottingham", help="maestro | midi_folder")
    parser.add_argument("--shuffle_data_set", type=str, default=True, help='')
    parser.add_argument('--nbworkers', type=int, default=3, help='')
    # Novel arguments
    parser.add_argument('--frame_bar', type=int, default=64, help='put a power of 2 here')
    parser.add_argument('--score_type', type=str, default='all', help='use mono measures or poly ones')
    parser.add_argument('--score_sig', type=str, default='all', help='rhythmic signature to use (use "all" to bypass)')
    # parser.add_argument('--data_keys',      type=str, default='C',      help='transpose all tracks to a given key')
    parser.add_argument('--data_normalize', type=int, default=1, help='normalize the data')
    parser.add_argument('--data_binarize', type=int, default=1, help='binarize the data')
    parser.add_argument('--data_pitch', type=int, default=1, help='constrain pitches in the data')
    parser.add_argument('--data_export', type=int, default=0, help='recompute the dataset (for debug purposes)')
    parser.add_argument('--data_augment', type=int, default=1, help='use data augmentation')
    # Parse the arguments
    args = parser.parse_args()
    # Data importing
    train_loader, valid_loader, test_loader, train_set, valid_set, test_set, args = import_dataset(args)
    # %%
    final_tr = trans.RandomApply(
        [trans.RandomChoice([Transpose(6), MaskColumns(), MaskRows(), TimeFlip(), PitchFlip()])], p=.5)
    batch = next(iter(train_loader))
