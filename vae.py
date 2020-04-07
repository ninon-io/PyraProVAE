import torch
from torch import nn
from torch.nn import functional as F


class BaseEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size, latent_size, batch_size, num_layers=2):
        super(BaseEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.LSTM = nn.LSTM(self.input_dim, self.hidden_size, self.num_layers, bidirectional=True)

        # Define output layer but maybe not
        self.linear = nn.Linear(self.hidden_size, latent_size)

    def init_hidden(self):
        # initialize the the hidden state
        return (torch.zero(self.num_layers, self.batch_size, self.hidden_size),
                torch.zero(self.num_layers, self.batch_size, self.hidden_size))

    def forward(self, x, h0, c0):
        batch_size = x.shape[0]
        _, (h, _) = self.LSTM(x, (h0, c0))
        h = h.view(self.num_layers, 2, batch_size, -1)
        h = h[-1]
        h = torch.cat([h[0], h[1]], dim=1)
        return h


class BaseDecoder(nn.Module):
    def __init__(self, input_size, latent_size, cond_hidden_size, cond_outdim):




class VaeModel(nn.Module):
    def __init__(self, batch_size, channel):
        super(VaeModel, self).__init__()
        self.batch_size = batch_size
        self.channel = channel


