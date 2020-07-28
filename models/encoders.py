# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
import random
import numpy as np
from models.layers import GatedDense, ResConv2d, ResConvTranspose2d


# -----------------------------------------------------------
# -----------------------------------------------------------
#
# Encoder section
#
# -----------------------------------------------------------
# -----------------------------------------------------------

class Encoder(nn.Module):

    def __init__(self, input_size, enc_size, args):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.enc_size = enc_size

    def forward(self, x, ctx=None):
        out = []
        return out

    def init(self, vals):
        pass


# -----------------------------------------------------------
#
# Very basic MLP encoder
#
# -----------------------------------------------------------

class EncoderMLP(nn.Module):

    def __init__(self, args, n_layers=5, **kwargs):
        super(EncoderMLP, self).__init__(**kwargs)
        type_mod = 'normal'
        in_size = args.input_size[0] * args.input_size[1]
        hidden_size = args.enc_hidden_size * 2
        out_size = args.enc_hidden_size
        dense_module = (type_mod == 'gated') and GatedDense or nn.Linear
        # Create modules
        modules = nn.Sequential()
        for l in range(n_layers):
            in_s = (l == 0) and in_size or hidden_size
            out_s = (l == n_layers - 1) and out_size or hidden_size
            modules.add_module('l%i' % l, dense_module(in_s, out_s))
            if l < n_layers - 1:
                modules.add_module('b%i' % l, nn.BatchNorm1d(out_s))
                modules.add_module('a%i' % l, nn.LeakyReLU())
                modules.add_module('a%i' % l, nn.Dropout(p=.3))
        self.net = modules
        self.init_parameters()

    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for m in self.net:
            if m.__class__ in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
                init.normal_(m.weight.data, mean=1, std=0.02)
                init.constant_(m.bias.data, 0)
            elif m.__class__ in [nn.Linear]:
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)
        # self.net[-1].weight.data.uniform_(-0.001, 0.001)
        # self.net[-1].bias.data.uniform_(-0.001, 0.001)

    def forward(self, x, ctx=None):
        # Flatten the input
        out = x.contiguous().view(x.shape[0], -1)
        for m in range(len(self.net)):
            out = self.net[m](out)
        return torch.tanh(out)


# -----------------------------------------------------------
#
# Basic CNN Encoder
#
# -----------------------------------------------------------


class EncoderCNN(nn.Module):

    def __init__(self, args, channels=64, n_layers=5, n_mlp=3):
        super(EncoderCNN, self).__init__()
        conv_module = (args.type_mod == 'residual') and ResConv2d or nn.Conv2d
        dense_module = (args.type_mod == 'residual') and GatedDense or nn.Linear
        # Create modules
        modules = nn.Sequential()
        size = [args.input_size[1], args.input_size[0]]
        out_size = args.enc_hidden_size
        hidden_size = args.enc_hidden_size
        in_channel = 1 if len(args.input_size) < 3 else args.input_size[0]  # in_size is (C,H,W) or (H,W)
        kernel = [4, 13]
        stride = [1, 1]
        """ First do a CNN """
        for l in range(n_layers):
            dil = 1
            pad = 2
            in_s = (l == 0) and in_channel or channels
            out_s = (l == n_layers - 1) and 1 or channels
            modules.add_module('c2%i' % l, conv_module(in_s, out_s, kernel, stride, pad, dilation=dil))
            if l < n_layers - 1:
                modules.add_module('b2%i' % l, nn.BatchNorm2d(out_s))
                modules.add_module('a2%i' % l, nn.ReLU())
                modules.add_module('d2%i' % l, nn.Dropout2d(p=.25))
            size[0] = int((size[0] + 2 * pad - (dil * (kernel[0] - 1) + 1)) / stride[0] + 1)
            size[1] = int((size[1] + 2 * pad - (dil * (kernel[1] - 1) + 1)) / stride[1] + 1)
        self.net = modules
        self.mlp = nn.Sequential()
        """ Then go through MLP """
        for l in range(n_mlp):
            in_s = (l == 0) and (size[0] * size[1]) or hidden_size
            out_s = (l == n_mlp - 1) and out_size or hidden_size
            self.mlp.add_module('h%i' % l, dense_module(in_s, out_s))
            if l < n_layers - 1:
                self.mlp.add_module('b%i' % l, nn.BatchNorm1d(out_s))
                self.mlp.add_module('a%i' % l, nn.LeakyReLU())
                self.mlp.add_module('d%i' % l, nn.Dropout(p=.25))
        self.cnn_size = size
        self.init_parameters()

    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for net in [self.net, self.mlp]:
            for m in net:
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

    def forward(self, inputs):
        out = inputs.unsqueeze(1) if len(inputs.shape) < 4 else inputs  # force to (batch, C, H, W)
        for m in range(len(self.net)):
            out = self.net[m](out)
        out = out.view(inputs.shape[0], -1)
        for m in range(len(self.mlp)):
            out = self.mlp[m](out)
        return torch.tanh(out)


# -----------------------------------------------------------
#
# GRU Encoder
#
# -----------------------------------------------------------

class EncoderGRU(nn.Module):

    def __init__(self, args):
        super(EncoderGRU, self).__init__()
        self.gru_0 = nn.GRU(
            args.input_size[0],
            args.enc_hidden_size,
            batch_first=True,
            bidirectional=True)
        self.linear_enc = nn.Linear(args.enc_hidden_size * 2, args.enc_hidden_size)
        self.bn_enc = nn.BatchNorm1d(args.enc_hidden_size)
        self.init_parameters()

    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for m in self.modules():
            if m.__class__ in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
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

    def forward(self, x, ctx=None):
        self.gru_0.flatten_parameters()
        x = self.gru_0(x)
        x = x[-1]
        x = x.transpose_(0, 1).contiguous()
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.bn_enc(self.linear_enc(x)))
        return x


# -----------------------------------------------------------
#
# CNN-GRU Encoder
#
# -----------------------------------------------------------

class EncoderCNNGRU(nn.Module):

    def __init__(self, args, channels=64, n_layers=5):
        super(EncoderCNNGRU, self).__init__()
        conv_module = (args.type_mod == 'residual') and ResConv2d or nn.Conv2d
        # First go through a CNN
        modules = nn.Sequential()
        size = [args.input_size[1], args.input_size[0]]
        in_channel = 1 if len(args.input_size) < 3 else args.input_size[0]  # in_size is (C,H,W) or (H,W)
        kernel = [4, 13]
        stride = [1, 1]
        """ First do a CNN """
        for l in range(n_layers):
            dil = 1
            pad = 2
            in_s = (l == 0) and in_channel or channels
            out_s = (l == n_layers - 1) and 1 or channels
            modules.add_module('c2%i' % l, conv_module(in_s, out_s, kernel, stride, pad, dilation=dil))
            if l < n_layers - 1:
                modules.add_module('b2%i' % l, nn.BatchNorm2d(out_s))
                modules.add_module('a2%i' % l, nn.ReLU())
                modules.add_module('d2%i' % l, nn.Dropout2d(p=.25))
            size[0] = int((size[0] + 2 * pad - (dil * (kernel[0] - 1) + 1)) / stride[0] + 1)
            size[1] = int((size[1] + 2 * pad - (dil * (kernel[1] - 1) + 1)) / stride[1] + 1)
        self.net = modules
        self.gru_0 = nn.GRU(
            size[1],
            args.enc_hidden_size,
            batch_first=True,
            bidirectional=True)
        self.cnn_size = size
        self.linear_enc = nn.Linear(args.enc_hidden_size * 2, args.enc_hidden_size)
        self.bn_enc = nn.BatchNorm1d(args.enc_hidden_size)
        self.init_parameters()

    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for m in self.modules():
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

    def forward(self, x, ctx=None):
        out = x.unsqueeze(1)
        self.gru_0.flatten_parameters()
        for m in range(len(self.net)):
            out = self.net[m](out)
        x = self.gru_0(out.squeeze(1))
        x = x[-1]
        x = x.transpose_(0, 1).contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_enc(self.linear_enc(x)))
        return x


# -----------------------------------------------------------
#
# Hierarchical encoder based on MusicVAE
#
# -----------------------------------------------------------

class EncoderHierarchical(nn.Module):

    def __init__(self, args):
        super(EncoderHierarchical, self).__init__()
        self.enc_hidden_size = args.enc_hidden_size
        self.latent_size = args.latent_size
        self.num_layers = args.num_layers
        self.device = args.device

        # Define the LSTM layer
        self.RNN = nn.LSTM(args.input_size[0], args.enc_hidden_size, batch_first=True, num_layers=args.num_layers,
                           bidirectional=True, dropout=0.4)
        self.linear_enc = nn.Linear(args.enc_hidden_size * 2, args.enc_hidden_size)
        self.bn_enc = nn.BatchNorm1d(args.enc_hidden_size)
        self.init_parameters()

    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for m in self.modules():
            if m.__class__ in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
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

    def init_hidden(self, batch_size=1):
        # initialize the the hidden state // Bidirectionnal so num_layers * 2 \\
        return (
            torch.zeros(self.num_layers * 2, batch_size, self.enc_hidden_size, dtype=torch.float, device=self.device),
            torch.zeros(self.num_layers * 2, batch_size, self.enc_hidden_size, dtype=torch.float, device=self.device))

    def forward(self, x, ctx=None):
        self.RNN.flatten_parameters()
        x, _ = self.RNN(x)
        x = x[-1]
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.bn_enc(self.linear_enc(x)))
        return x


# -----------------------------------------------------------
# -----------------------------------------------------------
#
# Decoder section
#
# -----------------------------------------------------------
# -----------------------------------------------------------


class Decoder(nn.Module):

    def __init__(self, in_size, out_size, args):
        super(Decoder, self).__init__()
        self.dec_size = in_size
        self.output_size = out_size

    def forward(self, x, ctx=None):
        out = []
        return out

    def init(self, vals):
        pass


# -----------------------------------------------------------
#
# Very basic MLP decoder
#
# -----------------------------------------------------------

class DecoderMLP(nn.Module):

    def __init__(self, args, n_layers=6, **kwargs):
        super(DecoderMLP, self).__init__(**kwargs)
        type_mod = 'normal'
        in_size = args.latent_size
        hidden_size = args.enc_hidden_size * 2
        out_size = args.input_size[0] * args.input_size[1] * args.num_classes
        self.output_size = args.input_size
        dense_module = (type_mod == 'gated') and GatedDense or nn.Linear
        # Create modules
        modules = nn.Sequential()
        for l in range(n_layers):
            in_s = (l == 0) and in_size or hidden_size
            out_s = (l == n_layers - 1) and out_size or hidden_size
            modules.add_module('l%i' % l, dense_module(in_s, out_s))
            if l < n_layers - 1:
                modules.add_module('b%i' % l, nn.BatchNorm1d(out_s))
                modules.add_module('a%i' % l, nn.LeakyReLU())
                modules.add_module('a%i' % l, nn.Dropout(p=.3))
        self.net = modules
        self.num_classes = args.num_classes
        self.init_parameters()

    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for m in self.net:
            if m.__class__ in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
                init.normal_(m.weight.data, mean=1, std=0.02)
                init.constant_(m.bias.data, 0)
            elif m.__class__ in [nn.Linear]:
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)
        # self.net[-1].weight.data.uniform_(-0.001, 0.001)
        # self.net[-1].bias.data.uniform_(-0.001, 0.001)

    def forward(self, z, ctx=None):
        # Flatten the input
        out = z.view(z.shape[0], -1)
        for m in range(len(self.net)):
            out = self.net[m](out)
        if self.num_classes > 1:
            out = F.log_softmax(out.view(z.size(0), self.output_size[1], self.num_classes, -1), 2)
        out = out.view(z.size(0), self.output_size[1], -1)
        return out


# -----------------------------------------------------------
#
# Basic CNN decoder
#
# -----------------------------------------------------------


class DecoderCNN(nn.Module):

    def __init__(self, args, channels=64, n_layers=5, n_mlp=2):
        super(DecoderCNN, self).__init__()
        conv_module = (args.type_mod == 'residual') and ResConvTranspose2d or nn.ConvTranspose2d
        dense_module = (args.type_mod == 'residual') and GatedDense or nn.Linear
        # Create modules
        cnn_size = [args.cnn_size[0], args.cnn_size[1]]
        self.cnn_size = cnn_size
        size = args.cnn_size
        kernel = [4, 13]
        stride = [1, 1]
        in_size = args.latent_size
        hidden_size = args.dec_hidden_size
        self.mlp = nn.Sequential()
        out_size = [args.num_classes, args.input_size[1], args.input_size[0]]
        """ First go through MLP """
        for l in range(n_mlp):
            in_s = (l == 0) and in_size or hidden_size
            out_s = (l == n_mlp - 1) and np.prod(cnn_size) or hidden_size
            self.mlp.add_module('h%i' % l, dense_module(in_s, out_s))
            if l < n_layers - 1:
                self.mlp.add_module('b%i' % l, nn.BatchNorm1d(out_s))
                self.mlp.add_module('a%i' % l, nn.ReLU())
                self.mlp.add_module('d%i' % l, nn.Dropout(p=.25))
        modules = nn.Sequential()
        """ Then do a CNN """
        for l in range(n_layers):
            dil = 1
            pad = 2
            out_pad = (pad % 2)
            in_s = (l == 0) and 1 or channels
            out_s = (l == n_layers - 1) and out_size[0] or channels
            modules.add_module('c2%i' % l,
                               conv_module(in_s, out_s, kernel, stride, pad, output_padding=out_pad, dilation=dil))
            if l < n_layers - 1:
                modules.add_module('b2%i' % l, nn.BatchNorm2d(out_s))
                modules.add_module('a2%i' % l, nn.ReLU())
                modules.add_module('a2%i' % l, nn.Dropout2d(p=.25))
            size[0] = int((size[0] - 1) * stride[0] - (2 * pad) + dil * (kernel[0] - 1) + out_pad + 1)
            size[1] = int((size[1] - 1) * stride[1] - (2 * pad) + dil * (kernel[1] - 1) + out_pad + 1)
        self.net = modules
        self.out_size = out_size  # (H,W) or (C,H,W)
        self.num_classes = args.num_classes
        self.init_parameters()

    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for net in [self.net, self.mlp]:
            for m in net:
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

    def forward(self, inputs):
        out = inputs
        for m in range(len(self.mlp)):
            out = self.mlp[m](out)
        out = out.unsqueeze(1).view(-1, 1, self.cnn_size[0], self.cnn_size[1])
        for m in range(len(self.net)):
            out = self.net[m](out)
        if len(self.out_size) < 3 or self.num_classes < 2:
            out = out[:, :, :self.out_size[0], :self.out_size[1]].squeeze(1)
        else:
            out = F.log_softmax(out[:, :, :self.out_size[1], :self.out_size[2]], 1)
            out = out.transpose(1, 2).contiguous().view(out.shape[0], self.out_size[1], -1)
        return out


# -----------------------------------------------------------
#
# GRU decoder
#
# -----------------------------------------------------------

class DecoderGRU(nn.Module):

    def __init__(self, args, k=500):
        super(DecoderGRU, self).__init__()
        self.grucell_1 = nn.GRUCell(
            args.latent_size + (args.input_size[0] * args.num_classes),
            args.dec_hidden_size)
        self.grucell_2 = nn.GRUCell(args.dec_hidden_size, args.dec_hidden_size)
        self.linear_init_1 = nn.Linear(args.latent_size, args.dec_hidden_size)
        self.linear_out_1 = nn.Linear(args.dec_hidden_size, args.input_size[0] * args.num_classes)
        self.k = torch.FloatTensor([k])
        self.eps = 1
        self.iteration = 0
        self.n_step = args.input_size[1]
        self.input_size = args.input_size[0]
        self.num_classes = args.num_classes
        self.init_parameters()

    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for m in self.modules():
            if m.__class__ in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
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

    def _sampling(self, x):
        if self.num_classes > 1:
            idx = x.view(x.shape[0], self.num_classes, -1).max(1)[1]
            x = F.one_hot(idx, num_classes=self.num_classes)
        return x.view(x.shape[0], -1)

    def forward(self, z):
        out = torch.zeros((z.size(0), (self.input_size * self.num_classes)))
        out[:, -1] = 1.
        x, hx = [], [None, None]
        t = torch.tanh(self.linear_init_1(z))
        hx[0] = t
        out = out.to(z.device)
        for i in range(self.n_step):
            out = torch.cat([out.float(), z], 1)
            hx[0] = self.grucell_1(out, hx[0])
            if i == 0:
                hx[1] = hx[0]
            hx[1] = self.grucell_2(hx[0], hx[1])
            tmp_out = self.linear_out_1(hx[1])
            if self.num_classes > 1:
                out = F.log_softmax(tmp_out.view(z.size(0), self.num_classes, -1), 1).view(z.size(0), -1)
            x.append(out)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    out = self.sample[:, i, :]
                else:
                    out = self._sampling(out)
                self.eps = self.k / \
                           (self.k + torch.exp(float(self.iteration) / self.k))
            else:
                out = self._sampling(out)
        return torch.stack(x, 1)


# -----------------------------------------------------------
#
# CNN-GRU decoder
#
# -----------------------------------------------------------

class DecoderCNNGRU(nn.Module):

    def __init__(self, args, k=500, channels=64, n_layers=5):
        super(DecoderCNNGRU, self).__init__()
        self.grucell_1 = nn.GRUCell(
            args.latent_size + (args.input_size[0] * args.num_classes),
            args.dec_hidden_size)
        self.grucell_2 = nn.GRUCell(args.dec_hidden_size, args.dec_hidden_size)
        self.linear_init_1 = nn.Linear(args.latent_size, args.dec_hidden_size)
        self.linear_out_1 = nn.Linear(args.dec_hidden_size, args.input_size[0] * args.num_classes)
        self.k = torch.FloatTensor([k])
        self.eps = 1
        self.iteration = 0
        self.n_step = args.input_size[1]
        self.input_size = args.input_size[0]
        self.num_classes = args.num_classes
        self.init_parameters()
        conv_module = (args.type_mod == 'residual') and ResConvTranspose2d or nn.ConvTranspose2d
        # Go through a CNN after RNN
        modules = nn.Sequential()
        cnn_size = [args.cnn_size[0], args.cnn_size[1]]
        self.cnn_size = cnn_size
        size = args.cnn_size
        self.linear_out_2 = nn.Linear(args.dec_hidden_size, self.cnn_size[0] * self.cnn_size[1])  # TODO
        kernel = [4, 13]
        stride = [1, 2]
        out_size = [args.num_classes, args.input_size[1], args.input_size[0]]
        for layer in range(n_layers):
            dil = 1
            pad = 2
            out_pad = (pad % 2)
            in_s = (layer == 0) and 1 or channels
            out_s = (layer == n_layers - 1) and out_size[0] or channels
            modules.add_module('c2%i' % layer, conv_module(in_s, out_s, kernel, stride, pad, dilation=dil))
            if layer < n_layers - 1:
                modules.add_module('b2%i' % layer, nn.BatchNorm2d(out_s))
                modules.add_module('a2%i' % layer, nn.ReLU())
                modules.add_module('d2%i' % layer, nn.Dropout2d(p=.25))
            size[0] = int((size[0] - 1) * stride[0] - 2 * pad + dil * (kernel[0] - 1) + out_pad + 1)
            size[1] = int((size[1] - 1) * stride[1] - 2 * pad - dil * (kernel[1] - 1) + out_pad + 1)
        self.net = modules
        self.out_size = out_size  # (H,W) or (C,H,W)

    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for m in self.modules():
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

    def _sampling(self, x):
        if self.num_classes > 1:
            idx = x.view(x.shape[0], self.num_classes, -1).max(1)[1]
            x = F.one_hot(idx, num_classes=self.num_classes)
        return x.view(x.shape[0], -1)

    def forward(self, z):
        out = torch.zeros((z.size(0), (self.input_size * self.num_classes)))
        out[:, -1] = 1.
        x, hx = [], [None, None]
        t = torch.tanh(self.linear_init_1(z))
        hx[0] = t
        out = out.to(z.device)
        for i in range(self.n_step):
            print('*' * 15)
            print('iteration:', i)
            print('out:', out.shape)
            print('z:', z.shape)
            out = torch.cat([out.float(), z], 1)
            hx[0] = self.grucell_1(out, hx[0])
            if i == 0:
                hx[1] = hx[0]
            hx[1] = self.grucell_2(hx[0], hx[1])
            tmp_out = self.linear_out_1(hx[1])
            print('tmp_out:', tmp_out.shape)
            if self.num_classes > 1:
                out = F.log_softmax(tmp_out.view(z.size(0), self.num_classes, -1), 1).view(z.size(0), -1)
            x.append(out)
            print('out before training:', out.shape)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    out = self.sample[:, i, :]
                else:
                    out = self._sampling(out)
                self.eps = self.k / \
                           (self.k + torch.exp(float(self.iteration) / self.k))
            else:
                out = self._sampling(out)
            print('final out:', out.shape)
        out = torch.stack(x, 1)
        out = out.view(-1, 512)
        print('after loop out:', out.shape)
        # out = out.reshape(-1, 1, self.cnn_size[0], self.cnn_size[1])
        print('*' * 15)
        # print('after view:', out.shape)
        out = self.linear_out_2(out)
        print('out after linear', out.shape)
        out = out.unsqueeze(1).view(-1, 1, self.cnn_size[0], self.cnn_size[1])
        print('out after unsqueeze', out.shape)
        for m in range(len(self.net)):
            out = self.net[m](out)
        if len(self.out_size) < 3 or self.num_classes < 2:
            out = out[:, :, :self.out_size[0], :self.out_size[1]].squeeze(1)
        else:
            out = F.log_softmax(out[:, :, :self.out_size[1], :self.out_size[2]], 1)
            out = out.transpose(1, 2).contiguous().view(out.shape[0], self.out_size[1], -1)
        return out


# -----------------------------------------------------------
#
# Hierarchical encoder based on MusicVAE
#
# -----------------------------------------------------------


class DecoderHierarchical(nn.Module):
    def __init__(self, args, k=500):
        super(DecoderHierarchical, self).__init__()
        self.device = args.device
        self.num_subsequences = args.num_subsequences
        self.input_size = args.input_size[0]
        self.cond_hidden_size = args.cond_hidden_size
        self.dec_hidden_size = args.dec_hidden_size
        self.num_layers = args.num_layers
        self.seq_length = args.input_size[1]
        self.subseq_size = self.seq_length // self.num_subsequences
        self.teacher_forcing_ratio = 0.5
        self.tanh = nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.fc_init_cond = nn.Linear(args.latent_size, args.cond_hidden_size * args.num_layers)
        self.conductor_RNN = nn.LSTM(args.latent_size // args.num_subsequences, args.cond_hidden_size, batch_first=True,
                                     num_layers=2,
                                     bidirectional=False, dropout=0.6)
        self.conductor_output = nn.Linear(args.cond_hidden_size, args.cond_output_dim)
        self.fc_init_dec = nn.Linear(args.cond_output_dim, args.dec_hidden_size * args.num_layers)
        self.decoder_RNN = nn.GRUCell(args.cond_output_dim + (self.input_size * args.num_classes),
                                      args.dec_hidden_size)  # , batch_first=True,
        # num_layers=2,
        # bidirectional=False, dropout=0.6)
        self.decoder_output = nn.Linear(args.dec_hidden_size, self.input_size * args.num_classes)
        self.num_classes = args.num_classes
        self.k = torch.FloatTensor([k])
        self.eps = 1
        self.iteration = 0
        self.init_parameters()

    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for m in self.modules():
            if m.__class__ in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
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

    def _sampling(self, x):
        if self.num_classes > 1:
            idx = x.view(x.shape[0], self.num_classes, -1).max(1)[1]
            x = F.one_hot(idx, num_classes=self.num_classes)
        return x.view(x.shape[0], -1)

    def forward(self, latent):
        batch_size = latent.shape[0]
        # Get the initial state of the conductor
        h0_cond = self.tanh(self.fc_init_cond(latent)).view(self.num_layers, batch_size, -1).contiguous()
        # Divide the latent code in subsequences
        latent = latent.view(batch_size, self.num_subsequences, -1)
        # Pass through the conductor
        subseq_embeddings, _ = self.conductor_RNN(latent, (h0_cond, h0_cond))
        subseq_embeddings = self.conductor_output(subseq_embeddings)
        # Get the initial states of the decoder
        h0s_dec = self.tanh(self.fc_init_dec(subseq_embeddings)).view(self.num_layers, batch_size,
                                                                      self.num_subsequences, -1).contiguous()
        # init the output seq and the first token to 0 tensors
        out = []
        token = torch.zeros(batch_size, (self.input_size * self.num_classes), dtype=torch.float, device=self.device)
        # autoregressivly output tokens
        for sub in range(self.num_subsequences):
            subseq_embedding = subseq_embeddings[:, sub, :]
            h0_dec = torch.mean(h0s_dec[:, :, sub, :].contiguous(), 0)
            for i in range(self.subseq_size):
                # Concat the previous token and the current sub embedding as input
                dec_input = torch.cat((token.float(), subseq_embedding), 1)
                # Pass through the decoder
                h0_dec = self.decoder_RNN(dec_input, h0_dec)
                token = self.decoder_output(h0_dec)
                if self.num_classes > 1:
                    token = F.log_softmax(token.view(latent.size(0), self.num_classes, -1), 1).view(latent.size(0), -1)
                # Fill the out tensor with the token
                out.append(token)
                if self.training:
                    p = torch.rand(1).item()
                    if p < self.eps:
                        token = self.sample[:, i, :]
                    else:
                        token = self._sampling(token)
                    self.eps = self.k / \
                               (self.k + torch.exp(float(self.iteration) / self.k))
                else:
                    token = self._sampling(token)
        return torch.stack(out, 1)
