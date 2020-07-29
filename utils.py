# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.init as init

#%% ---------------------------------------------------------
#
# Initialization utils
#
# -----------------------------------------------------------
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
                
#%% ---------------------------------------------------------
#
# Classification utils
#
# -----------------------------------------------------------


class LatentDataset(torch.utils.data.Dataset):
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
        print(out.shape)
        print(y)
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
