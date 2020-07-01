# -*- coding: utf-8 -*-
import torch
from torch import nn
import random
import numpy as np
from models.layers import GatedDense


# -----------------------------------------------------------
# -----------------------------------------------------------
#
# Constructor section
#
# -----------------------------------------------------------
# -----------------------------------------------------------


def construct_encoder_decoder(in_size, enc_size, latent_size, hidden_size=512, channels=32, n_layers=6, n_mlp=2,
                              type_ae='ae', type_mod='gated_cnn', args=None):
    # Construct encoder and decoder layers for AE models
    # MLP layers
    if type_mod in ['mlp', 'gated_mlp']:
        type_ed = (type_mod == 'mlp') and 'normal' or 'gated'
        encoder = GatedMLP(np.prod(in_size), enc_size, hidden_size, n_layers, type_ed)
        decoder = DecodeMLP(latent_size, in_size, hidden_size, n_layers, type_ed)
    elif type_mod in ['cnn', 'gated_cnn', 'res_cnn']:
        type_ed = (type_mod == 'cnn') and 'normal' or ((type_mod == 'res_cnn') and 'residual' or 'gated')
        encoder = GatedCNN(in_size, enc_size, channels, n_layers, hidden_size, n_mlp, type_ed, args)
        decoder = DecodeCNN(latent_size, encoder.cnn_size, in_size, channels, n_layers, hidden_size, n_mlp, type_ed,
                            args)
    return encoder, decoder


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

    def __init__(self, in_size, out_size, hidden_size=512, n_layers=6, type_mod='gated', **kwargs):
        super(GatedMLP, self).__init__(**kwargs)
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
# Hierarchical encoder based on MusicVAE
#
# -----------------------------------------------------------


class HierarchicalEncoder(Encoder):
    def __init__(self, input_size, enc_size, args):
        # Initialize the encoder
        super(HierarchicalEncoder, self).__init__(input_size, enc_size, args)
        self.enc_hidden_size = args.enc_hidden_size
        self.latent_size = args.latent_size
        self.num_layers = args.num_layers
        self.device = args.device

        # Define the LSTM layer
        self.RNN = nn.LSTM(self.input_size, self.enc_size, batch_first=True, num_layers=args.num_layers,
                           bidirectional=True, dropout=0.6)

    def init_hidden(self, batch_size=1):
        # Initialize the the hidden state // Bidirectionnal so num_layers * 2 \\
        return (
            torch.zeros(self.num_layers * 2, batch_size, self.enc_hidden_size, dtype=torch.float, device=self.device),
            torch.zeros(self.num_layers * 2, batch_size, self.enc_hidden_size, dtype=torch.float, device=self.device))

    def forward(self, x, ctx=None):
        h0, c0 = ctx
        batch_size = x.shape[0]
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

    def __init__(self, in_size, out_size, hidden_size=512, n_layers=6, type_mod='gated', **kwargs):
        super(DecodeMLP, self).__init__(in_size, np.prod(out_size), hidden_size, n_layers, type_mod, **kwargs)
        # Record final size
        self.out_size = out_size

    def forward(self, x):
        # Use super function
        out = GatedMLP.forward(self, x)
        # Reshape output
        out = out.view(x.shape[0], *self.out_size)
        return out


# -----------------------------------------------------------
#
# Hierarchical encoder based on MusicVAE
#
# -----------------------------------------------------------


class HierarchicalDecoder(Decoder):
    def __init__(self, in_size, out_size, args):
        super(HierarchicalDecoder, self).__init__(in_size, out_size, args)
        self.device = args.device
        self.tanh = nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.fc_init_cond = nn.Linear(args.latent_size, args.cond_hidden_size * args.num_layers)
        self.conductor_RNN = nn.LSTM(args.latent_size // args.num_subsequences, args.cond_hidden_size, batch_first=True,
                                     num_layers=2,
                                     bidirectional=False, dropout=0.6)
        self.conductor_output = nn.Linear(args.cond_hidden_size, args.cond_output_dim)
        self.fc_init_dec = nn.Linear(args.cond_output_dim, args.dec_hidden_size * args.num_layers)
        self.decoder_RNN = nn.LSTM(args.cond_output_dim + args.input_size, args.dec_hidden_size, batch_first=True,
                                   num_layers=2,
                                   bidirectional=False, dropout=0.6)
        self.decoder_output = nn.Linear(args.dec_hidden_size, args.input_size)
        self.num_subsequences = args.num_subsequences
        self.input_size = args.input_size
        self.cond_hidden_size = args.cond_hidden_size
        self.dec_hidden_size = args.dec_hidden_size
        self.num_layers = args.num_layers
        self.seq_length = args.seq_length
        self.teacher_forcing_ratio = 0.5

    def forward(self, latent, target=None, teacher_forcing=None):
        batch_size = latent.shape[0]
        subseq_size = self.seq_length // self.num_subsequences
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
        token = torch.zeros(batch_size, subseq_size, self.input_size, dtype=torch.float, device=self.device)
        # autoregressivly output tokens
        for sub in range(self.num_subsequences):
            subseq_embedding = subseq_embeddings[:, sub, :]
            h0_dec = h0s_dec[:, :, sub, :].contiguous()
            c0_dec = h0s_dec[:, :, sub, :].contiguous()
            # Concat the previous token and the current sub embedding as input
            dec_input = torch.cat((token, subseq_embedding.unsqueeze(1).expand(-1, subseq_size, -1)), -1)
            # Pass through the decoder
            token, (h0_dec, c0_dec) = self.decoder_RNN(dec_input, (h0_dec, c0_dec))
            token = self.decoder_output(token)
            # Fill the out tensor with the token
            out[:, sub * subseq_size:((sub + 1) * subseq_size), :] = token
            # If teacher_forcing replace the output token by the real one sometimes
            if teacher_forcing:
                if random.random() <= self.teacher_forcing_ratio:
                    token = target[:, sub * subseq_size:((sub + 1) * subseq_size), :]
        return out


# -----------------------------------------------------------
#
# Simplified decoder for piano-roll
#
# -----------------------------------------------------------


class PianoRollDecoder(Decoder):
    def __init__(self, args):
        super(Decoder, self).__init__()
        # self.latent_to_conductor = nn.Linear(latent_size, latent_size)
        self.device = args.device
        self.tanh = nn.Tanh()
        self.conductor_RNN = nn.LSTM(args.latent_size, args.cond_hidden_size, batch_first=True, num_layers=2,
                                     bidirectional=False)
        self.conductor_output = nn.Linear(args.cond_hidden_size, args.cond_outdim)
        self.decoder_RNN = nn.LSTM(args.cond_outdim + args.input_size, args.hidden_size, batch_first=True, num_layers=2,
                                   bidirectional=False)
        self.decoder_output = nn.Linear(args.hidden_size, args.input_size)
        self.softmax = nn.Softmax()
        self.num_subsequences = args.num_subsequences
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.seq_length = args.seq_length
        self.teacher_forcing_ratio = 0.5

    def forward(self, latent, target=None, teacher_forcing=None, args=None):
        batch_size = latent.shape[0]
        target = torch.nn.functional.one_hot(target.long(), 389).float()
        h0, c0 = self.init_hidden(batch_size)
        out = torch.zeros(batch_size, self.seq_length, self.input_size, dtype=torch.float, device=args.device)
        prev_note = torch.zeros(batch_size, 1, self.input_size, dtype=torch.float, device=args.device)
        for subseq_idx in range(self.num_subsequences):
            subseq_embedding, (h0, c0) = self.conductor_RNN(latent.unsqueeze(1), (h0, c0))
            subseq_embedding = self.tanh(self.conductor_output(subseq_embedding))
            # Initialize lower decoder hidden state
            h0_dec = (torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=args.device),
                      torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=args.device))

            use_teacher_forcing = False  # if random.random() < self.teacher_forcing_ratio else False
            for note_idx in range(int(self.seq_length / self.num_subsequences)):
                e = torch.cat((prev_note, subseq_embedding), -1)
                prev_note, h0_dec = self.decoder_RNN(e, h0_dec)
                prev_note = self.tanh(self.decoder_output(prev_note))

                idx = subseq_idx * self.seq_length / self.num_subsequences + note_idx
                out[:, int(idx), :] = prev_note.squeeze()
                if use_teacher_forcing:
                    prev_note = target[:, int(idx), :].unsqueeze(1)
        return out

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=self.device))
