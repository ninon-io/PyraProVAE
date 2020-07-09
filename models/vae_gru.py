# https://github.com/cdyrhjohn/Deep-Music-Analogy-Demos/blob/master/code/model.py

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


class VAE(nn.Module):
    def __init__(self,
                 roll_dims,
                 hidden_dims,
                 rhythm_dims,
                 condition_dims,
                 z1_dims,
                 z2_dims,
                 n_step,
                 k=1000):
        super(VAE, self).__init__()
        self.gru_0 = nn.GRU(
            roll_dims + condition_dims,
            hidden_dims,
            batch_first=True,
            bidirectional=True)
        self.linear_mu = nn.Linear(hidden_dims * 2, z1_dims + z2_dims)
        self.linear_var = nn.Linear(hidden_dims * 2, z1_dims + z2_dims)
        self.grucell_0 = nn.GRUCell(z2_dims + rhythm_dims,
                                    hidden_dims)
        self.grucell_1 = nn.GRUCell(
            z1_dims + roll_dims + rhythm_dims + condition_dims, hidden_dims)
        self.grucell_2 = nn.GRUCell(hidden_dims, hidden_dims)
        self.linear_init_0 = nn.Linear(z2_dims, hidden_dims)
        self.linear_out_0 = nn.Linear(hidden_dims, rhythm_dims)
        self.linear_init_1 = nn.Linear(z1_dims, hidden_dims)
        self.linear_out_1 = nn.Linear(hidden_dims, roll_dims)
        self.n_step = n_step
        self.roll_dims = roll_dims
        self.hidden_dims = hidden_dims
        self.eps = 1
        self.rhythm_dims = rhythm_dims
        self.sample = None
        self.rhythm_sample = None
        self.iteration = 0
        self.z1_dims = z1_dims
        self.z2_dims = z2_dims
        self.k = torch.FloatTensor([k])

    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

    def encoder(self, x, condition):
        # self.gru_0.flatten_parameters()
        x = torch.cat((x, condition), -1)
        x = self.gru_0(x)[-1]
        x = x.transpose_(0, 1).contiguous()
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        var = self.linear_var(x).exp_()
        distribution_1 = Normal(mu[:, :self.z1_dims], var[:, :self.z1_dims])
        distribution_2 = Normal(mu[:, self.z1_dims:], var[:, self.z1_dims:])
        return distribution_1, distribution_2

    def rhythm_decoder(self, z):
        out = torch.zeros((z.size(0), self.rhythm_dims))
        out[:, -1] = 1.
        x = []
        t = torch.tanh(self.linear_init_0(z))
        hx = t
        if torch.cuda.is_available():
            out = out.cuda()
        for i in range(self.n_step):
            out = torch.cat([out, z], 1)
            hx = self.grucell_0(out, hx)
            out = F.log_softmax(self.linear_out_0(hx), 1)
            x.append(out)
            if self.training:
                p = torch.rand(1).item()
                print(self.eps)
                if p < self.eps:
                    out = self.rhythm_sample[:, i, :]
                else:
                    out = self._sampling(out)
            else:
                out = self._sampling(out)
        return torch.stack(x, 1)

    def final_decoder(self, z, rhythm, condition):
        out = torch.zeros((z.size(0), self.roll_dims))
        out[:, -1] = 1.
        x, hx = [], [None, None]
        t = torch.tanh(self.linear_init_1(z))
        hx[0] = t
        if torch.cuda.is_available():
            out = out.cuda()
        for i in range(self.n_step):
            out = torch.cat([out, rhythm[:, i, :], z, condition[:, i, :]], 1)
            hx[0] = self.grucell_1(out, hx[0])
            if i == 0:
                hx[1] = hx[0]
            hx[1] = self.grucell_2(hx[0], hx[1])
            out = F.log_softmax(self.linear_out_1(hx[1]), 1)
            x.append(out)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    out = self.sample[:, i, :]
                else:
                    out = self._sampling(out)
                self.eps = self.k / \
                    (self.k + torch.exp(self.iteration / self.k))
            else:
                out = self._sampling(out)
        return torch.stack(x, 1)

    def decoder(self, z1, z2, condition=None):
        rhythm = self.rhythm_decoder(z2)
        return self.final_decoder(z1, rhythm, condition)

    def forward(self, x, condition):
        if self.training:
            self.sample = x
            self.rhythm_sample = x[:, :, :-2].sum(-1).unsqueeze(-1)
            self.rhythm_sample = torch.cat((self.rhythm_sample, x[:, :, -2:]),
                                           -1)
            self.iteration += 1
        dis1, dis2 = self.encoder(x, condition)
        z1 = dis1.rsample()
        z2 = dis2.rsample()
        recon_rhythm = self.rhythm_decoder(z2)
        recon = self.final_decoder(z1, recon_rhythm, condition)
        output = (recon, recon_rhythm, dis1.mean, dis1.stddev, dis2.mean,
                  dis2.stddev)
        return output

class VAEKawai(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_dims,
                 z_dims,
                 n_step,
                 device,
                 num_classes,
                 k=400):
        super(VAEKawai, self).__init__()
        self.gru_0 = nn.GRU(
            vocab_size,
            hidden_dims,
            batch_first=True,
            bidirectional=True)
        self.linear_mu = nn.Linear(hidden_dims * 2, z_dims)
        self.linear_var = nn.Linear(hidden_dims * 2, z_dims)
        self.grucell_1 = nn.GRUCell(
            z_dims + (vocab_size * num_classes),
            hidden_dims)
        self.grucell_2 = nn.GRUCell(hidden_dims, hidden_dims)
        self.linear_init_1 = nn.Linear(z_dims, hidden_dims)
        self.linear_out_1 = nn.Linear(hidden_dims, vocab_size * num_classes)
        self.n_step = n_step
        self.vocab_size = vocab_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.eps = 1
        self.sample = None
        self.iteration = 0
        self.z_dims = z_dims
        self.k = torch.FloatTensor([k])
        self.device = device

    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

    def encoder(self, x):
        self.gru_0.flatten_parameters()
        x = self.gru_0(x)
        x = x[-1]
        x = x.transpose_(0, 1).contiguous()
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        var = self.linear_var(x).exp_()
        distribution = Normal(mu, var)
        return distribution, mu, var

    def encode(self, x):
        b, c, s = x.size()
        x = x.reshape(b, -1)
        x = torch.eye(self.vocab_size)[x].to(self.device)
        dis = self.encoder(x)
        z = dis.rsample()
        return z, mu, var

    def decoder(self, z):
        out = torch.zeros((z.size(0), (self.vocab_size * self.num_classes)))
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

    def forward(self, x):
        b, c, s = x.size()
        x = x.transpose(1, 2)
        #x = x.reshape(b, -1)
        x_indices = x
        #x = torch.eye(self.vocab_size)[x].to(self.device)
        if self.training:
            if (self.num_classes > 1):
                self.sample = torch.nn.functional.one_hot(x.long())
                self.sample = self.sample.view(b, s, -1)
            else:
                self.sample = x
            self.iteration += 1
        dis, mu, var = self.encoder(x)
        z = dis.rsample()
        recon = self.decoder(z)
        #preds = torch.argmax(recon, dim=-1)
        #loss = F.nll_loss(recon.reshape(-1, recon.size(-1)), x_indices.reshape(-1))
        recon = recon.transpose(1, 2)
        if (self.num_classes > 1):
            recon = recon.view(b, self.num_classes, self.vocab_size, -1)
        return mu, var, z, recon
