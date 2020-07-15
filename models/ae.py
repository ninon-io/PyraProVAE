import torch
import torch.nn as nn


class AE(nn.Module):

    def __init__(self, encoder_size, latent_size, args):
        super(AE, self).__init__()
        self.encoder, self.decoder = construct_encoder_decoder(args)
        self.latent_dims = latent_size
        self.map_latent = nn.Linear(encoder_size, latent_size)
        self.apply(self.init_parameters)

    def init_parameters(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, z):
        return self.decoder(z)

    def regularize(self, z):
        z = self.map_latent(z)
        return z, torch.zeros(z.shape[0]).to(z.device).mean()

    def forward(self, x):
        # Encode the inputs
        z = self.encode(x)
        # Potential regularization
        z_tilde, z_loss = self.regularize(z)
        # Decode the samples
        x_tilde = self.decode(z_tilde)
        return x_tilde, z_tilde, z_loss
