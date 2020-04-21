import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from vae import VaeModel


class Learn:
    def __init__(self, train_loader, test_loader, test_set, batch_size=512, seed=1, lr=0.01):
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VaeModel().to(device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.n_epochs = 1
        self.epoch = 0
        self.epoch_test = 0
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.test_set = test_set
        self.loss_mean = 0
        self.recon_loss_mean = 0
        self.kl_div_mean = 0
        self.w_kl = 0
        self.loss_mean_test = 0
        self.kl_div_mean_test = 0
        self.recon_loss_mean_test = 0

    def train(self, x):
        self.model.train() # to confirm here
        h_enc, c_enc = self.model[0].init_hidden(self.batch_size)
        mu, log_var = self.model[0](x, h_enc, c_enc)
        with torch.no_grad():
            latent = mu + log_var * torch.randn_like(mu)
        h_dec, c_dec = self.model[1].init_hidden(self.batch_size)
        x_recon = self.model[1](latent, x, h_dec, c_dec, teacher_forcing=True)
        kl_div = - 1/2 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        recon_loss = F.mse_loss(x_recon.squeeze(1), x)
        self.recon_loss_mean += recon_loss
        self.kl_div_mean += kl_div

        # Training pass
        loss = recon_loss + self.w_kl * kl_div
        self.loss_mean += loss
        self.optimizer.zero_grad()
        # Learning with back-propagation
        loss.backward()
        # Optimizes weights
        self.optimizer.step()

        return self.loss_mean, self.kl_div_mean, self.recon_loss_mean

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for i, V in tqdm(enumerate(self.test_loader), total=len(self.test_set)//self.batch_size):
                if V.max() != 0:
                    x = V / V.max()
                if torch.cuda.is_available():
                    x = x.cuda()
                # Training test pass
                h_enc, c_enc = self.model[0].init_hidden(self.batch_size)
                mu, log_var = self.model[0](x, h_enc, c_enc)
                latent = mu + log_var * torch.randn_like(mu)
                h_dec, c_dec = self.model[1].init_hidden(self.batch_size)
                x_recon = self.model[1](latent, x, h_dec, c_dec, teacher_forcing=True)
                kl_div = - 1 / 2 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                recon_loss = F.mse_loss(x_recon.squeeze(1), x)
                self.recon_loss_mean_test += recon_loss
                self.kl_div_mean_test += kl_div
                loss = recon_loss + self.w_kl * kl_div
                self.loss_mean_test += loss
                self.epoch_test = +1



















