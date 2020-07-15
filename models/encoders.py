# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
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

class GatedMLP(Encoder):

    def __init__(self, args, n_layers=5, **kwargs):
        super(GatedMLP, self).__init__(**kwargs)
        type_mod = 'normal'
        in_size = torch.cumprod(args.input_size)
        hidden_size = args.enc_hidden_size
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
                modules.add_module('a%i' % l, nn.ReLU())
                modules.add_module('a%i' % l, nn.Dropout(p=.3))
        self.net = modules

    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)

    def forward(self, x, ctx=None):
        # Flatten the input
        out = x.view(x.shape[0], -1)
        for m in range(len(self.net)):
            out = self.net[m](out)
        return out

# -----------------------------------------------------------
#
# Basic CNN Encoder
#
# -----------------------------------------------------------


class GatedCNN(Encoder):
    
    def __init__(self, args, channels = 32, n_layers = 5, hidden_size = 512, n_mlp = 3):
        super(GatedCNN, self).__init__()
        conv_module = (type_mod == 'residual') and ResConv2d or nn.Conv2d
        # Create modules
        modules = nn.Sequential()
        size = [in_size[-2], in_size[-1]]
        in_channel = 1 if len(in_size)<3 else in_size[0] #in_size is (C,H,W) or (H,W)
        kernel = args.kernel
        stride = 2
        """ First do a CNN """
        for l in range(n_layers):
            dil = ((args.dilation == 3) and (2 ** l) or args.dilation)
            pad = 3 * (dil + 1)
            in_s = (l==0) and in_channel or channels
            out_s = (l == n_layers - 1) and 1 or channels
            modules.add_module('c2%i'%l, conv_module(in_s, out_s, kernel, stride, pad, dilation = dil))
            if (l < n_layers - 1):
                modules.add_module('b2%i'%l, nn.BatchNorm2d(out_s))
                modules.add_module('a2%i'%l, nn.ReLU())
                modules.add_module('d2%i'%l, nn.Dropout2d(p=.25))
            size[0] = int((size[0]+2*pad-(dil*(kernel-1)+1))/stride+1)
            size[1] = int((size[1]+2*pad-(dil*(kernel-1)+1))/stride+1)
        self.net = modules
        self.mlp = nn.Sequential()
        """ Then go through MLP """
        for l in range(n_mlp):
            in_s = (l==0) and (size[0] * size[1]) or hidden_size
            out_s = (l == n_mlp - 1) and out_size or hidden_size
            self.mlp.add_module('h%i'%l, dense_module(in_s, out_s))
            if (l < n_layers - 1):
                self.mlp.add_module('b%i'%l, nn.BatchNorm1d(out_s))
                self.mlp.add_module('a%i'%l, nn.ReLU())
                self.mlp.add_module('d%i'%l, nn.Dropout(p=.25))
        self.cnn_size = size
    
    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)
        
    def forward(self, inputs):
        out = inputs.unsqueeze(1) if len(inputs.shape) < 4 else inputs # force to (batch, C, H, W)
        for m in range(len(self.net)):
            out = self.net[m](out)
        out = out.view(inputs.shape[0], -1)
        for m in range(len(self.mlp)):
            out = self.mlp[m](out)
        return out

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
        
    def forward(self, x, ctx=None):
        self.gru_0.flatten_parameters()
        x = self.gru_0(x)
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

class HierarchicalEncoder(Encoder):
    def __init__(self, input_size, enc_size, args):
        super(HierarchicalEncoder, self).__init__(input_size, enc_size, args)
        self.enc_hidden_size = args.enc_hidden_size
        self.latent_size = args.latent_size
        self.num_layers = args.num_layers
        self.device = args.device

        # Define the LSTM layer
        self.RNN = nn.LSTM(args.input_size[0], args.enc_hidden_size, batch_first=True, num_layers=args.num_layers,
                           bidirectional=True, dropout=0.6)

    def init_hidden(self, batch_size=1):
        # initialize the the hidden state // Bidirectionnal so num_layers * 2 \\
        return (torch.zeros(self.num_layers * 2, batch_size, self.enc_hidden_size, dtype=torch.float, device=self.device),
                torch.zeros(self.num_layers * 2, batch_size, self.enc_hidden_size, dtype=torch.float, device=self.device))

    def forward(self, x, ctx=None):
        h0, c0 = ctx
        batch_size = x.shape[0]
        x = x.transpose(1, 2)
        _, (h, _) = self.RNN(x, (h0, c0))
        h = h.view(self.num_layers, 2, batch_size, -1)
        h = h[-1]
        h = torch.cat([h[0], h[1]], dim=1)
        return h


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

class DecodeMLP(Decoder):

    def __init__(self, in_size, out_size, args):
        super(DecodeMLP, self).__init__(in_size, out_size, args)
        # Record final size
        self.out_size = out_size

    def forward(self, x, ctx=None):
        # Use super function
        out = GatedMLP.forward(self, x)
        # Reshape output
        out = out.view(x.shape[0], *self.out_size)
        return out

# -----------------------------------------------------------
#
# Basic CNN decoder
#
# -----------------------------------------------------------

    
class DecodeCNN(Decoder):
    
    def __init__(self, in_size, cnn_size, out_size, channels = 32, n_layers = 5, hidden_size = 512, n_mlp = 2, type_mod='gated', args=None):
        super(DecodeCNN, self).__init__()
        conv_module = (type_mod == 'residual') and ResConvTranspose2d or nn.ConvTranspose2d
        dense_module = (type_mod == 'gated') and GatedDense or nn.Linear
        # Create modules
        self.cnn_size = [cnn_size[0], cnn_size[1]]
        size = cnn_size
        kernel = args.kernel
        stride = 2
        self.mlp = nn.Sequential()
        """ First go through MLP """
        for l in range(n_mlp):
            in_s = (l==0) and (in_size) or hidden_size
            out_s = (l == n_mlp - 1) and np.prod(cnn_size) or hidden_size
            self.mlp.add_module('h%i'%l, dense_module(in_s, out_s))
            if (l < n_layers - 1):
                self.mlp.add_module('b%i'%l, nn.BatchNorm1d(out_s))
                self.mlp.add_module('a%i'%l, nn.ReLU())
                self.mlp.add_module('d%i'%l, nn.Dropout(p=.25))
        modules = nn.Sequential()
        """ Then do a CNN """
        for l in range(n_layers):
            dil = ((args.dilation == 3) and (2 ** ((n_layers - 1) - l)) or args.dilation)
            pad = 3 * (dil + 1)
            if (args.dilation == 1):
                pad = 2
            out_pad = (pad % 2)
            in_s = (l==0) and 1 or channels
            out_s = (l == n_layers - 1) and out_size[0] or channels
            modules.add_module('c2%i'%l, conv_module(in_s, out_s, kernel, stride, pad, output_padding=out_pad, dilation = dil))
            if (l < n_layers - 1):
                modules.add_module('b2%i'%l, nn.BatchNorm2d(out_s))
                modules.add_module('a2%i'%l, nn.ReLU())
                modules.add_module('a2%i'%l, nn.Dropout2d(p=.25))
            size[0] = int((size[0] - 1) * stride - (2 * pad) + dil * (kernel - 1) + out_pad + 1)
            size[1] = int((size[1] - 1) * stride - (2 * pad) + dil * (kernel - 1) + out_pad + 1)
        self.net = modules
        self.out_size = out_size #(H,W) or (C,H,W)
    
    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)
        
    def forward(self, inputs):
        out = inputs
        for m in range(len(self.mlp)):
            out = self.mlp[m](out)
        out = out.unsqueeze(1).view(-1, 1, self.cnn_size[0], self.cnn_size[1])
        for m in range(len(self.net)):
            out = self.net[m](out)
        if len(self.out_size) < 3:
            out = out[:, :, :self.out_size[0], :self.out_size[1]].squeeze(1)
        else:
            out = out[:, :, :self.out_size[1], :self.out_size[2]]
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
    
    def _sampling(self, x):
        if (self.num_classes > 1):
            idx = x.view(x.shape[0], self.num_classes, -1).max(1)[1]
            x = F.one_hot(idx, num_classes = self.num_classes)
        return x.view(x.shape[0], -1)
    
    def forward(self, z):
        out = torch.zeros((z.size(0), (self.input_size * self.num_classes)))
        out[:, -1] = 1.
        x, hx = [], [None, None]
        t = torch.tanh(self.linear_init_1(z))
        hx[0] = t
        if torch.cuda.is_available():
            out = out.cuda()
        for i in range(self.n_step):
            out = torch.cat([out.float(), z], 1)
            hx[0] = self.grucell_1(out, hx[0])
            if i == 0:
                hx[1] = hx[0]
            hx[1] = self.grucell_2(hx[0], hx[1])
            out = F.log_softmax(self.linear_out_1(hx[1]).view(z.size(0), self.num_classes, -1), 1).view(z.size(0), -1)
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
# Hierarchical encoder based on MusicVAE
#
# -----------------------------------------------------------


class HierarchicalDecoder(Decoder):
    def __init__(self, in_size, out_size, args):
        super(HierarchicalDecoder, self).__init__(in_size, out_size, args)
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
        self.decoder_RNN = nn.LSTM(args.cond_output_dim + self.input_size, args.dec_hidden_size, batch_first=True,
                                   num_layers=2,
                                   bidirectional=False, dropout=0.6)
        self.decoder_output = nn.Linear(args.dec_hidden_size, self.input_size)

    def forward(self, latent, target=None, teacher_forcing=None):
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
        out = torch.zeros(batch_size, self.seq_length, self.input_size, dtype=torch.float, device=self.device)
        token = torch.zeros(batch_size, self.subseq_size, self.input_size, dtype=torch.float, device=self.device)
        # autoregressivly output tokens
        for sub in range(self.num_subsequences):
            subseq_embedding = subseq_embeddings[:, sub, :]
            h0_dec = h0s_dec[:, :, sub, :].contiguous()
            c0_dec = h0s_dec[:, :, sub, :].contiguous()
            # Concat the previous token and the current sub embedding as input
            dec_input = torch.cat((token, subseq_embedding.unsqueeze(1).expand(-1, self.subseq_size, -1)), -1)
            # Pass through the decoder
            token, (h0_dec, c0_dec) = self.decoder_RNN(dec_input, (h0_dec, c0_dec))
            token = self.decoder_output(token)
            # Fill the out tensor with the token
            out[:, sub * self.subseq_size:((sub + 1) * self.subseq_size), :] = token
            # If teacher_forcing replace the output token by the real one sometimes
            if teacher_forcing:
                if random.random() <= self.teacher_forcing_ratio:
                    token = target[:, :, sub * self.subseq_size:((sub + 1) * self.subseq_size)].transpose(1, 2)
        out = out.transpose(1, 2)
        return out
