import torch
import torch.nn as nn
from torch.distributions import Normal

# -----------------------------------------------------------
# -----------------------------------------------------------
#
# Basic auto-encoder (no regularization)
#
# -----------------------------------------------------------
# -----------------------------------------------------------

class AE(nn.Module):

    def __init__(self, encoder, decoder, args):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder.iteration = 0
        self.num_classes = args.num_classes
        self.input_size = args.input_size[0]
        self.map_latent = nn.Linear(args.enc_hidden_size, args.latent_size)
        self.loss = torch.Tensor(1).zero_().to(args.device)

    def encode(self, x):
        # Re-arrange to put time first
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = self.map_latent(x)
        return x

    def decode(self, z):
        recon = self.decoder(z)
        recon = recon.transpose(1, 2)
        if self.num_classes > 1:
            recon = recon.view(z.shape[0], self.num_classes, self.input_size, -1)
        return recon

    def forward(self, x):
        b, c, s = x.size()
        if self.training:
            if self.num_classes > 1:
                self.sample = torch.nn.functional.one_hot(x.long())
                self.sample = self.sample.view(b, s, -1)
            else:
                self.sample = x
            self.decoder.sample = self.sample
            self.decoder.iteration += 1
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z, self.loss

# -----------------------------------------------------------
# -----------------------------------------------------------
#
# Variational auto-encoder (Kullback-Leibler regularization)
#
# -----------------------------------------------------------
# -----------------------------------------------------------
    
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
        # Re-arrange to put time first
        x = x.transpose(1, 2)
        out = self.encoder(x)
        mu = self.linear_mu(out)
        var = self.linear_var(out).exp_()
        distribution = Normal(mu, var)
        z = distribution.rsample()
        return z, mu, var
    
    def regularize(self, z, mu, var):
        n_batch = z.shape[0]
        # Compute KL divergence
        kl_div = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)
        # Normalize by size of batch
        # kl_div = kl_div / n_batch
        return kl_div
    
    def decode(self, z):
        recon = self.decoder(z)
        recon = recon.transpose(1, 2)
        if self.num_classes > 1:
            recon = recon.view(z.shape[0], self.num_classes, self.input_size, -1)
        return recon

    def forward(self, x):
        b, c, s = x.size()
        if self.training:
            if self.num_classes > 1:
                self.sample = torch.nn.functional.one_hot(x.long())
                self.sample = self.sample.view(b, s, -1)
            else:
                self.sample = x
            self.decoder.sample = self.sample
            self.decoder.iteration += 1
        # Encode the inut
        z, mu, var = self.encode(x)
        # Regularize the latent
        loss = self.regularize(z, mu, var)
        # Perform decoding
        recon = self.decode(z)
        return recon, z, loss

# -----------------------------------------------------------
# -----------------------------------------------------------
#
# Wasserstein auto-encoder (Maximum Mean Discrepancy regularization)
#
# -----------------------------------------------------------
# -----------------------------------------------------------

class WAE(VAE):

    def __init__(self, encoder, decoder, args):
        super(WAE, self).__init__(encoder, decoder, args)

    def regularize(self, z, mu, var):
        n_batch = z.size(0)
        # Re-parametrize
        q = Normal(torch.zeros(mu.shape[1]), torch.ones(var.shape[1]))
        # Sample from the z prior
        z_prior = q.sample((n_batch,)).to(z.device)
        # Compute MMD divergence
        mmd_dist = compute_mmd(z, z_prior)
        return z, mmd_dist

# -----------------------------------------------------------
#
# WAE helper functions
#
# -----------------------------------------------------------
        
def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd

# -----------------------------------------------------------
# -----------------------------------------------------------
#
# Variational auto-encoder with flows (latent regularization)
#
# -----------------------------------------------------------
# -----------------------------------------------------------

class VAEFlow(VAE):

    def __init__(self, encoder, decoder, flow, input_dims, encoder_dims, latent_dims):
        super(VAEFlow, self).__init__(encoder, decoder, input_dims, encoder_dims, latent_dims)
        self.flow_enc = nn.Linear(encoder_dims, flow.n_parameters())
        self.flow = flow
        self.apply(self.init_parameters)

    def init_parameters(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            m.weight.data.uniform_(-0.01, 0.01)
            m.bias.data.fill_(0.0)

    def encode(self, x):
        out = self.encoder(x)
        mu = self.linear_mu(out)
        var = self.linear_var(out).exp_()
        distribution = Normal(mu, var)
        return distribution, mu, var
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        mu = self.mu(x)
        log_var = self.log_var(x)
        flow_params = self.flow_enc(x)
        return mu, log_var, flow_params

    def latent(self, x, z_params):
        n_batch = x.size(0)
        # Split the encoded values to retrieve flow parameters
        mu, log_var, flow_params = z_params
        # Re-parametrize a Normal distribution
        # q = distrib.Normal(torch.zeros(mu.shape[1]), torch.ones(log_var.shape[1]))
        # eps = q.sample((n_batch, )).detach().to(x.device)
        eps = torch.randn_like(mu).detach().to(x.device)
        # Obtain our first set of latent points
        z_0 = (log_var.exp().sqrt() * eps) + mu
        # Update flows parameters
        self.flow.set_parameters(flow_params)
        # Complexify posterior with flows
        z_k, list_ladj = self.flow(z_0)
        # ln p(z_k)
        log_p_zk = torch.sum(-0.5 * z_k * z_k, dim=1)
        # ln q(z_0)  (not averaged)
        log_q_z0 = torch.sum(-0.5 * (log_var + (z_0 - mu) * (z_0 - mu) * log_var.exp().reciprocal()), dim=1)
        # ln q(z_0) - ln p(z_k)
        logs = (log_q_z0 - log_p_zk).sum()
        # print([p.mean() for p in list_ladj])
        # Add log determinants
        ladj = torch.cat(list_ladj, dim=1)
        # print('Flow')
        # print(torch.sum(log_q_z0))
        # print(torch.sum(log_p_zk))
        # print(torch.sum(ladj))
        # ln q(z_0) - ln p(z_k) - sum[log det]
        logs -= torch.sum(ladj)
        # print(logs)
        return z_k, (logs / float(n_batch))