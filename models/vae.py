import torch
import numpy as np
import torch.nn as nn
import torch.distributions as distrib
from models.ae import AE
from models.encoders import construct_encoder_decoder


class VAE(AE):

    def __init__(self, input_dims, encoder_size, latent_size, args, gaussian_dec=False):
        super(VAE, self).__init__(encoder_size, latent_size, args)
        self.encoder, self.decoder = construct_encoder_decoder(args)
        self.input_dims = input_dims
        self.latent_dims = latent_size
        self.encoder_size = encoder_size
        # Latent gaussians
        self.mu = nn.Linear(encoder_size, latent_size)
        self.log_var = nn.Sequential(nn.Linear(encoder_size, latent_size))
        # Gaussian decoder
        if gaussian_dec:
            in_prod = np.prod(input_dims)
            self.mu_dec = nn.Linear(in_prod, in_prod)
            self.log_var_dec = nn.Sequential(
                nn.Linear(in_prod, in_prod))
        self.gaussian_dec = gaussian_dec
        self.apply(self.init_parameters)

    def init_parameters(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var

    def decode(self, z):
        if self.gaussian_dec:
            n_batch = z.size(0)
            x_vals = self.decoder(z)
            x_vals = x_vals.view(-1, np.prod(self.input_dims))
            mu = self.mu_dec(x_vals)
            log_var = self.log_var_dec(x_vals)
            q = distrib.Normal(torch.zeros(mu.shape[1]), torch.ones(log_var.shape[1]))
            eps = q.sample((n_batch,)).detach().to(z.device)
            x_tilde = (log_var.exp().sqrt() * eps) + mu
            x_tilde = x_tilde.view(-1, *self.input_dims)
        else:
            x_tilde = self.decoder(z)
        return x_tilde

    def forward(self, x):
        # Encode the inputs
        z_params = self.encode(x)
        # z_q_mean, z_q_logvar = z_params
        # Obtain latent samples and latent loss
        z_tilde, z_loss = self.latent(x, z_params)
        # Decode the samples
        x_tilde = self.decode(z_tilde)
        return x_tilde, z_tilde, z_loss  # , z_q_mean, z_q_logvar

    def latent(self, x, z_params):
        n_batch = x.size(0)
        # Retrieve mean and var
        mu, log_var = z_params
        # Re-parametrize
        # q = distrib.Normal(torch.zeros(mu.shape[1]), torch.ones(log_var.shape[1]))
        # eps = q.sample((n_batch, )).detach().to(x.device)
        eps = torch.randn_like(mu).detach().to(x.device)
        z = (log_var.exp().sqrt() * eps) + mu
        # Compute KL divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_div = kl_div / n_batch
        return z, kl_div


class VaeModelPyraPro(AE):
    def __init__(self, encoder_size, latent_size, args, teacher_forcing=True):
        super(VaeModelPyraPro, self).__init__(encoder_size, latent_size, args)
        self.encoder_size = encoder_size
        self.tf = teacher_forcing
        self.device = args.device
        self.encoder, self.decoder = construct_encoder_decoder(args)
        self.hidden_to_mu = nn.Linear(2 * self.encoder.enc_hidden_size, self.encoder.latent_size)
        self.hidden_to_sigma = nn.Linear(2 * self.encoder.enc_hidden_size, self.encoder.latent_size)

    def forward(self, x):
        # Encoder pass
        batch_size = x.size(0)
        h_enc, c_enc = self.encoder.init_hidden(batch_size)
        hidden = self.encoder(x, h_enc, c_enc)
        # Reparameterization
        mu = self.hidden_to_mu(hidden)
        sigma = self.hidden_to_sigma(hidden)
        eps = torch.randn_like(mu).detach().to(x.device)
        latent = (sigma.exp().sqrt() * eps) + mu
        # Decoder pass
        x_reconstruct = self.decoder(latent, x, teacher_forcing=self.tf)
        return mu, sigma, latent, x_reconstruct

    # Generate bar from latent space
    def generate(self, latent):
        # Create dumb target
        input_shape = (1, self.decoder.seq_length, self.decoder.input_size)
        trg = torch.zeros(input_shape)
        # Forward pass in the decoder
        generated_bar = self.decoder(latent.unsqueeze(0), trg, teacher_forcing=False)

        return generated_bar
