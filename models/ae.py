import torch
import torch.nn as nn
from torch.distributions import Normal

class AE(nn.Module):

    def __init__(self, encoder, decoder, args):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder.iteration = 0
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
        b, c, s = x.size()
        # Re-arrange to put time first
        x = x.transpose(1, 2)
        if self.training:
            if self.num_classes > 1:
                self.sample = torch.nn.functional.one_hot(x.long())
                self.sample = self.sample.view(b, s, -1)
            else:
                self.sample = x
            self.decoder.sample = self.sample
            self.decoder.iteration += 1
        z = self.encoder(x)
        recon = self.decoder(z)
        recon = recon.transpose(1, 2)
        if self.num_classes > 1:
            recon = recon.view(b, self.num_classes, self.input_size, -1)
        return z, z, z, recon


class VAE(nn.Module):
    def __init__(self, encoder, decoder, args):
        super(VAE, self).__init__()
        self.sample = None
        self.encoder = encoder
        self.decoder = decoder
        self.decoder.iteration = 0
        self.num_classes = args.num_classes
        self.input_size = args.input_size[0]
        self.linear_mu = nn.Linear(args.enc_hidden_size, args.latent_size)
        self.linear_var = nn.Linear(args.enc_hidden_size, args.latent_size)

    # Generate bar from latent space
    def generate(self, z):
        # Forward pass in the decoder
        generated_bar = self.decoder(z.unsqueeze(0))
        return generated_bar

    def encode(self, x):
        out = self.encoder(x)
        mu = self.linear_mu(out)
        var = self.linear_var(out).exp_()
        distribution = Normal(mu, var)
        return distribution, mu, var

    def forward(self, x):
        b, c, s = x.size()
        # Re-arrange to put time first
        x = x.transpose(1, 2).contiguous()
        if self.training:
            if self.num_classes > 1:
                self.sample = torch.nn.functional.one_hot(x.long())
                self.sample = self.sample.view(b, s, -1)
            else:
                self.sample = x
            self.decoder.sample = self.sample
            self.decoder.iteration += 1
        dis, mu, var = self.encode(x)
        z = dis.rsample()
        recon = self.decoder(z)
        recon = recon.transpose(1, 2).contiguous()
        if self.num_classes > 1:
            recon = recon.view(b, self.num_classes, self.input_size, -1)
        return mu, var, z, recon