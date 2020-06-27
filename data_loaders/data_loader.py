import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import math
import pretty_midi
from statistics import mean
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional
from guppy import hpy
import argparse
from texttable import Texttable

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

    # Find the perfect match transforms, transforms.functional.center_crop((0, 0), (128, 100, 0))
    # def __getitem__(self, index):
    #     transform = transforms.Compose([transforms.Normalize([0.5], [0.5])])
    #     sample = torch.load(self.bar_dir + '/' + self.bar_files[index])
    #     norm_sample = transform(sample.unsqueeze(0))
    #     return norm_sample

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


def maximum(train_set, valid_set, test_set):  # TODO: transfer to np
    t = Texttable()
    # Compute the maximum of the dataset
    global_track = []
    for x in train_set:
        global_track.append(x)
    for y in valid_set:
        global_track.append(y)
    for z in test_set:
        global_track.append(z)
    max_global = torch.max(torch.stack(global_track))
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
    # print(10 * '*******')
    # print('Casual information on your beautiful dataset with normalization:')
    # t.add_rows([['', 'Maximum', 'Minimum', 'Mean', 'Std', 'Var', 'NaN', 'Inf'],
    #             ['Train', torch.max(torch.stack(track_train)),
    #              torch.min(torch.stack(track_train)),
    #              torch.mean(torch.stack(track_train)),
    #              torch.std(torch.stack(track_train)),
    #              torch.var(torch.stack(track_train)),
    #              torch.isnan(torch.stack(track_train)).byte().any(),
    #              torch.isinf(torch.stack(track_train)).byte().any()],
    #             ['Validate', torch.max(torch.stack(track_valid)),
    #              torch.min(torch.stack(track_valid)),
    #              torch.mean(torch.stack(track_valid)),
    #              torch.std(torch.stack(track_valid)),
    #              torch.var(torch.stack(track_valid)),
    #              torch.isnan(torch.stack(track_valid)).byte().any(),
    #              torch.isinf(torch.stack(track_valid)).byte().any()],
    #             ['Test', torch.max(torch.stack(track_test)),
    #              torch.min(torch.stack(track_test)),
    #              torch.mean(torch.stack(track_test)),
    #              torch.std(torch.stack(track_test)),
    #              torch.var(torch.stack(track_test)),
    #              torch.isnan(torch.stack(track_test)).byte().any(),
    #              torch.isinf(torch.stack(track_test)).byte().any()]])
    # print(t.draw())
    # print('Maximum global before normalization:', max_global)
    # print(10 * '*******')
    return max_global, track_train, track_valid, track_test


# Main data import
def import_dataset(args):
    # Main transform
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=2.342, std=12.476)])  # Rescale?
    # Retrieve correct data loader
    if args.dataset == "maestro":  # Dataset is already splitted in 3 folders
        train_path = "/fast-1/mathieu/datasets/maestro_folders/train"
        test_path = "/fast-1/mathieu/datasets/maestro_folders/test"
        valid_path = "/fast-1/mathieu/datasets/maestro_folders/valid"
        train_set_raw = PianoRollRep(train_path, args.frame_bar, export=False)
        test_set_raw = PianoRollRep(test_path, args.frame_bar, export=False)
        valid_set_raw = PianoRollRep(valid_path, args.frame_bar, export=False)
        # Normalization
        max_global, train_set, valid_set, test_set = maximum(train_set_raw, valid_set_raw, test_set_raw)
        # train_set, valid_set, test_set = train_set_raw, valid_set_raw, test_set_raw  # No normalization
        # Get sampler
        train_indices, valid_indices = list(range(len(train_set))), list(range(len(valid_set)))
        train_indices = train_indices[:16000]  # TODO: DON'T FORGET TO DELETE THIS
        valid_indices = valid_indices[:256]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

    elif args.dataset == "nottingham":
        train_path = "/fast-1/mathieu/datasets/Nottingham/train"
        test_path = "/fast-1/mathieu/datasets/Nottingham/test"
        valid_path = "/fast-1/mathieu/datasets/Nottingham/valid"
        train_set_raw = PianoRollRep(train_path, args.frame_bar, export=False)
        test_set_raw = PianoRollRep(test_path, args.frame_bar, export=False)
        valid_set_raw = PianoRollRep(valid_path, args.frame_bar, export=False)
        # Normalization
        # max_global, train_set, valid_set, test_set = maximum(train_set_raw, valid_set_raw, test_set_raw)
        train_set, valid_set, test_set = train_set_raw, valid_set_raw, test_set_raw
        # Get sampler
        train_indices, valid_indices = list(range(len(train_set))), list(range(len(valid_set)))
        train_indices = train_indices[:1000]  # TODO: DON'T FORGET TO DELETE THIS
        valid_indices = valid_indices[:256]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

    elif args.dataset == "bach_chorales":
        train_path = "/fast-1/mathieu/datasets/JSB_Chorales/train"
        test_path = "/fast-1/mathieu/datasets/JSB_Chorales/test"
        valid_path = "/fast-1/mathieu/datasets/JSB_Chorales/valid"
        train_set_raw = PianoRollRep(train_path, args.frame_bar, export=False)
        test_set_raw = PianoRollRep(test_path, args.frame_bar, export=False)
        valid_set_raw = PianoRollRep(valid_path, args.frame_bar, export=False)
        # Normalization
        # max_global, train_set, valid_set, test_set = maximum(train_set_raw, valid_set_raw, test_set_raw)
        train_set, valid_set, test_set = train_set_raw, valid_set_raw, test_set_raw
        # Get sampler
        train_indices, valid_indices = list(range(len(train_set))), list(range(len(valid_set)))
        train_indices = train_indices[:16000]  # TODO: DON'T FORGET TO DELETE THIS
        valid_indices = valid_indices[:256]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataloader')
    # Device Information
    parser.add_argument('--device', type=str, default='cuda:0', help='device cuda or cpu')

    # Data Parameters
    parser.add_argument('--midi_path', type=str, default='/fast-1/mathieu/datasets/maestro_folders/train',
                        help='path to midi folder')
    parser.add_argument("--test_size", type=float, default=0.2, help="% of data used in test set")
    parser.add_argument("--valid_size", type=float, default=0.2, help="% of data used in valid set")
    parser.add_argument("--dataset", type=str, default="maestro", help="maestro | midi_folder")
    parser.add_argument("--shuffle_data_set", type=str, default=True, help='')

    # Model Saving and reconstruction
    parser.add_argument('--model_path', type=str, default='/slow-2/ninon/pyrapro/models_saving/entire_model/',
                        help='path to the saved model')
    parser.add_argument('--weights_path', type=str, default='/slow-2/ninon/pyrapro/models_saving/weights/',
                        help='path to the saved model')
    parser.add_argument('--figure_reconstruction_path', type=str, default='/slow-2/ninon/pyrapro/reconstruction/',
                        help='path to reconstruction figures')

    # Model Parameters
    parser.add_argument("--model", type=str, default="PyraPro", help='Name of the model')

    # PyraPro specific parameters: dimensions of the architecture
    parser.add_argument('--input_dim', type=int, default=100, help='do not touch if you do not know')
    parser.add_argument('--enc_hidden_size', type=int, default=2048, help='do not touch if you do not know')
    parser.add_argument('--latent_size', type=int, default=512, help='do not touch if you do not know')
    parser.add_argument('--cond_hidden_size', type=int, default=1024, help='do not touch if you do not know')
    parser.add_argument('--cond_output_dim', type=int, default=512, help='do not touch if you do not know')
    parser.add_argument('--dec_hidden_size', type=int, default=1024, help='do not touch if you do not know')
    parser.add_argument('--num_layers', type=int, default=2, help='do not touch if you do not know')
    parser.add_argument('--num_subsequences', type=int, default=8, help='do not touch if you do not know')
    parser.add_argument('--seq_length', type=int, default=128, help='do not touch if you do not know')

    # Optimization parameters
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='input batch size for testing')
    parser.add_argument('--frame_bar', type=int, default=100, help='correspond to input dim')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--nbworkers', type=int, default=3, help='')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    # Parse the arguments
    args = parser.parse_args()
    # Data importing
    train_loader, valid_loader, test_loader, train_set, valid_set, test_set, args = import_dataset(args)
