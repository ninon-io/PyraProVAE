import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


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


class EncoderCNNGRU(nn.Module):

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
        return x


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
        if self.num_classes > 1:
            idx = x.view(x.shape[0], self.num_classes, -1).max(1)[1]
            x = F.one_hot(idx, num_classes=self.num_classes)
        return x.view(x.shape[0], -1)

    # Generate bar from latent space (Unused)
    def generate(self, z):
        # Forward pass in the decoder
        generated_bar = self.decoder(z)
        return generated_bar

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
        # x = x.reshape(b, -1)
        x_indices = x
        # x = torch.eye(self.vocab_size)[x].to(self.device)
        if self.training:
            if self.num_classes > 1:
                self.sample = torch.nn.functional.one_hot(x.long())
                self.sample = self.sample.view(b, s, -1)
            else:
                self.sample = x
            self.iteration += 1
        dis, mu, var = self.encoder(x)
        z = dis.rsample()
        recon = self.decoder(z)
        # preds = torch.argmax(recon, dim=-1)
        # loss = F.nll_loss(recon.reshape(-1, recon.size(-1)), x_indices.reshape(-1))
        recon = recon.transpose(1, 2)
        if self.num_classes > 1:
            recon = recon.view(b, self.num_classes, self.vocab_size, -1)
        return mu, var, z, recon
