# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.init as init


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