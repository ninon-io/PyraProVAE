import torch
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np

from vae import VaeModel
from vae import HierarchicalDecoder, HierarchicalEncoder


class Learn:
    def __init__(self, train_loader, test_loader, train_set, test_set, batch_size=512, seed=1, lr=0.01):
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = 100
        self.enc_hidden_size = 2048
        self.latent_size = 512
        self.cond_hidden_size = 1024
        self.cond_output_dim = 512
        self.dec_hidden_size = 1024
        self.num_layers = 2
        self.num_subsequences = 8
        self.seq_length = 128
        self.encoder = HierarchicalEncoder(input_dim=self.input_dim, enc_hidden_size=self.enc_hidden_size,
                                           latent_size=self.latent_size)
        self.decoder = HierarchicalDecoder(input_size=self.input_dim, latent_size=self.latent_size,
                                           cond_hidden_size=self.cond_hidden_size, cond_outdim=self.cond_output_dim,
                                           dec_hidden_size=self.dec_hidden_size, num_layers=self.num_layers,
                                           num_subsequences=self.num_subsequences, seq_length=self.seq_length)
        self.model = VaeModel(encoder=self.encoder, decoder=self.decoder).double().to(device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.n_epochs = 1
        self.epoch = 0
        self.epoch_test = 0
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_set = train_set
        self.test_set = test_set
        self.loss_mean = 0
        self.recon_loss_mean = 0
        self.kl_div_mean = 0
        self.beta = 0
        self.loss_mean_test = 0
        self.kl_div_mean_test = 0
        self.recon_loss_mean_test = 0

    def train(self):
        self.model.train()
        for batch_idx, x in tqdm(enumerate(self.train_loader), total=len(self.train_set) // self.batch_size):
            x = x.to(self.device)
            h_enc, c_enc = self.encoder.init_hidden(batch_size=self.batch_size)  # TODO: NOT FUCKING WORKING
            mu, log_var = self.encoder(x.double(), h_enc, c_enc)
            with torch.no_grad():
                latent = mu + log_var * torch.randn_like(mu)
            h_dec, c_dec = self.decoder.init_hidden(self.batch_size)   # TODO: STILL NOT WORKING =(
            x_recon = self.decoder(latent, x, h_dec, c_dec, teacher_forcing=True)
            kl_div = - 1/2 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            recon_loss = F.mse_loss(x_recon.squeeze(1), x)
            self.recon_loss_mean += recon_loss
            self.kl_div_mean += kl_div
            # Training pass
            loss = recon_loss + self.beta * kl_div
            self.loss_mean += loss
            self.optimizer.zero_grad()
            # Learning with back-propagation
            loss.backward()
            # Optimizes weights
            self.optimizer.step()
            if self.n_epochs > 10 and self.beta < 1:
                self.beta += 0.0025
            self.n_epochs += 1

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for i, V in tqdm(enumerate(self.test_loader), total=len(self.test_set)//self.batch_size):
                if V.max() != 0:
                    x = V / V.max()
                x = x.to(self.device)
                # Training test pass
                # h_enc, c_enc = self.encoder.init_hidden(batch_size=self.batch_size)  # seems to work...?
                # mu, log_var = self.encoder(x.double(), h_enc, c_enc)
                # latent = mu + log_var * torch.randn_like(mu)
                # h_dec, c_dec = self.decoder.init_hidden(self.batch_size)  # same problem here...?
                # x_recon = self.model[1](latent, x, h_dec, c_dec, teacher_forcing=True)
                mu, sigma, latent, x_recon = self.model(x)
                log_var = np.log(sigma**2)
                kl_div = - 1 / 2 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                recon_loss = F.mse_loss(x_recon.squeeze(1), x)
                self.recon_loss_mean_test += recon_loss
                self.kl_div_mean_test += kl_div
                loss = recon_loss + self.beta * kl_div
                self.loss_mean_test += loss
                self.epoch_test += 1



















