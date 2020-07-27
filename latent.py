#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:25:06 2020

@author: esling
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import music21
import music21.features.jSymbolic as jSymbolic
from data_loaders.data_loader import import_dataset
from mido import MidiFile, MidiTrack
from utils import roll_to_track

#%% - Argument parsing
parser = argparse.ArgumentParser(description='PyraProVAE')
# Device Information
parser.add_argument('--device',         type=str, default='cpu',    help='device cuda or cpu')
# Data Parameters
parser.add_argument('--midi_path',      type=str, default='/Users/esling/Datasets/symbolic/', help='path to midi folder')
parser.add_argument("--test_size",      type=float, default=0.2,    help="% of data used in test set")
parser.add_argument("--valid_size",     type=float, default=0.2,    help="% of data used in valid set")
parser.add_argument("--dataset",        type=str, default="nottingham", help="maestro | nottingham | bach_chorales | midi_folder")
parser.add_argument("--shuffle_data_set", type=int, default=0,      help='')
# Novel arguments
parser.add_argument('--frame_bar',      type=int, default=64,       help='put a power of 2 here')
parser.add_argument('--score_type',     type=str, default='mono',   help='use mono measures or poly ones')
parser.add_argument('--score_sig',      type=str, default='4_4',    help='rhythmic signature to use (use "all" to bypass)')
parser.add_argument('--data_normalize', type=int, default=1,        help='normalize the data')
parser.add_argument('--data_binarize',  type=int, default=1,        help='binarize the data')
parser.add_argument('--data_pitch',     type=int, default=1,        help='constrain pitches in the data')
parser.add_argument('--data_export',    type=int, default=0,        help='recompute the dataset (for debug purposes)')
parser.add_argument('--data_augment',   type=int, default=1,        help='use data augmentation')
parser.add_argument('--subsample',      type=int, default=0,        help='train on subset')
parser.add_argument('--nbworkers',      type=int, default=3,        help='')
# Optimization arguments
parser.add_argument('--batch_size',     type=int, default=64, help='input batch size')
# Parse the arguments
args = parser.parse_args()
#%% Handle device
args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# Here we add the model path directly
model_path = 'output/nottingham_mono_1_2_1_vae_cnn-gru_128_1.0_512/models/_epoch_200.pth'

#%% ---------------------------------------------------------
#
# Load dataset and model
#
# -----------------------------------------------------------
# Data importing
print('[Importing dataset]')
train_loader, valid_loader, test_loader, train_set, valid_set, test_set, args = import_dataset(args)
# Recall minimum pitch
args.min_pitch = train_set.min_p
# Reload best performing model
model = torch.load(model_path, map_location=args.device)

#%%
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

def symbolic_features(x, args):
    batch_features = []
    for x_cur in x:
        # First create a MIDI version
        midi = MidiFile(type = 1)
        midi.tracks.append(MidiTrack(roll_to_track(x_cur, args.min_pitch)))
        midi.save('/tmp/track.mid')
        # Then transform to a Music21 stream
        stream = music21.converter.parse('/tmp/track.mid')
        feature_vals = {}
        # Number of notes
        nb_notes = 0
        for n in stream.parts[0]:
            if (n.isRest):
                continue
            nb_notes += 1
        feature_vals['nb_notes'] = nb_notes
        # Perform all desired jSymbolic extraction
        for key, val in features.items():
            if (val[0] is None):
                continue
            try:
                feature_vals[key] = val[0](stream).extract().vector[0]
            except:
                feature_vals[key] = 0      
        batch_features.append(feature_vals)
    return batch_features

def retrieve_z(model, loader, args):
    cpt = 0
    mu_set = []
    latent_set = []
    final_features = {}
    for f in features:
        final_features[f] = []
    for x in loader:
        # Send to device
        x = x.to(args.device, non_blocking=True)
        print(cpt)
        # Compute symbolic features on input
        feats = symbolic_features(x, args)
        # Encode into model
        latent = model.encode(x)
        if (latent.__class__ == tuple):
            latent = latent[0]
            mu = latent[1]
            print(mu.shape)
            mu_set.append(mu)
        # Add current latent
        latent_set.append(latent)
        for b in feats:
            for f in features:
                final_features[f].append(b[f])
        cpt += 1
    # Concatenate into vector
    final_latent = torch.cat(latent_set, dim = 0)
    if (len(mu_set) > 1):
        mu_set = torch.cat(mu_set, dim = 0)
    return final_latent, final_features, mu_set

latent_train, features_train, mus_train = retrieve_z(model, train_loader, args)
latent_valid, features_valid, mus_valid = retrieve_z(model, valid_loader, args)
latent_test, features_test, mus_test = retrieve_z(model, test_loader, args)

from sklearn import manifold, decomposition
from mpl_toolkits.mplot3d import Axes3D

def compute_projection(dataset, n_dims=3, plot=True):
    pca = decomposition.PCA(n_components=3)
    pca.fit(mus_dset.detach())
    X = pca.transform(mus_dset.detach())
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=30, azim=1)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=features['nb_notes'], cmap=plt.cm.nipy_spectral, edgecolor='k')
    plt.cla()
    tsne = manifold.TSNE(n_components=3)
    X = tsne.fit_transform(mus_dset.detach())
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=50, azim=100)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=features['nb_notes'], cmap=plt.cm.nipy_spectral, edgecolor='k')


#%% -----------------------------------------------------------
#
# Classification section
#
# -----------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    """ Simplest dataset for latent """
    def __init__(self, latent, labels):
        self.latent = latent
        self.labels = labels

    def __len__(self):
        return self.latent.shape[0]

    def __getitem__(self, index):
        # Load data and get label
        x = self.latent[index]
        y = self.labels[index]
        return x, y

def epoch_train(model, optimizer, criterion, loader, args):
    model.train()
    # Create mean loss
    loss_mean = torch.zeros(1).to(args.device)
    for x, y in loader:
        # Send to device
        x = x.to(args.device, non_blocking=True)
        # Pass into model
        out = model(x)
        # Compute reconstruction criterion
        loss = criterion(out, y) / y.shape[0]
        loss_mean += loss.detach()
        optimizer.zero_grad()
        # Learning with back-propagation
        loss.backward()
        # Optimizes weights
        optimizer.step()
    return loss_mean

def epoch_test(model, optimizer, criterion, loader, args):
    model.eval()
    # Create mean loss
    loss_mean = torch.zeros(1).to(args.device)
    with torch.no_grad():
        for x, y in loader:
            # Send to device
            x = x.to(args.device, non_blocking=True)
            # Pass into model
            out = model(x)
            # Compute reconstruction criterion
            loss = criterion(out, y) / y.shape[0]
            loss_mean += loss.detach()
    return loss_mean

# Classifier properties
latent_size = latent_train.shape[1]
hidden_size = 256
args.lr = 1e-3
args.epochs = 100
classification_targets = ['nb_notes', 'note_density', 'quality', 'range', 
     'pitch_variety', 'amount_arpeggiation', 'direction_motion']
# Loop through all the features
for target in classification_targets:
    # Number of classes
    if (features[target][1] == 'int'):
        nb_classes = max(features_train[target])
    elif (features[target][1] == 'float'):
        nb_classes = 1
    else:
        nb_classes = 2
    # Create dataset holders
    z_train_set = Dataset(latent_train, features_train[target])
    z_train_loader = torch.utils.data.DataLoader(z_train_set, batch_size=64)
    z_valid_set = Dataset(latent_valid, features_valid[target])
    z_valid_loader = torch.utils.data.DataLoader(z_valid_set, batch_size=64)
    z_test_set = Dataset(latent_test, features_test[target])
    z_test_loader = torch.utils.data.DataLoader(z_test_set, batch_size=64)
    # Create simple classifier
    classifier = nn.Sequential()
    classifier.add_module('l1', nn.Linear(latent_size, hidden_size))
    classifier.add_module('b1', nn.BatchNorm1d(hidden_size))
    classifier.add_module('r1', nn.LeakyReLU())
    classifier.add_module('l2', nn.Linear(hidden_size, nb_classes))
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-07, eps=1e-08)
    # Losses
    if (nb_classes > 1):
        criterion = nn.NLLLoss(reduction='sum')
    else:
        criterion = nn.MSELoss(reduction='sum')    
    # -----------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------
    print(f'[Starting training on {target}]')
    # Set best to infinity
    cur_best_valid = np.inf
    best_test = np.inf
    # Through the epochs
    for epoch in range(1, args.epochs + 1, 1):
        print(f"Epoch: {epoch}")
        # Training epoch
        loss_train = epoch_train(classifier, optimizer, criterion, z_train_loader, args)
        # Validate epoch
        loss_valid = epoch_test(classifier, optimizer, criterion, z_valid_loader, args)
        # Step for learning rate
        scheduler.step(loss_valid)
        # Test model
        loss_test = epoch_test(classifier, optimizer, criterion, z_test_loader, args)
        print(f'Train : {loss_train}')
        print(f'Valid : {loss_valid}')
        print(f'Test : {loss_test}')
        if (loss_valid < cur_best_valid):
            cur_best_valid = loss_valid
            best_test = loss_test
