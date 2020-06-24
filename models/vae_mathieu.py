import torch
from torch import nn
import random


class VAE_pianoroll(nn.Module):
    def __init__(self, encoder, decoder, teacher_forcing=True):
        super(VAE_pianoroll, self).__init__()
        self.tf = teacher_forcing
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_to_mu = nn.Linear(2 * encoder.hidden_size, encoder.latent_size)
        self.hidden_to_sig = nn.Linear(2 * encoder.hidden_size, encoder.latent_size)

    def forward(self, x):
        # Encoder pass
        batch_size = x.size(0)
        h_enc, c_enc = self.encoder.init_hidden(batch_size)
        hidden = self.encoder(x, h_enc, c_enc)
        # Reparametrization
        mu = self.hidden_to_mu(hidden)
        sig = self.hidden_to_sig(hidden)
        eps = torch.randn_like(mu).detach().to(x.device)
        latent = (sig.exp().sqrt() * eps) + mu
        # Decoder pass
        x_reconst = self.decoder(latent, x, teacher_forcing=self.tf)
        return mu, sig, latent, x_reconst

    def generate(self, latent):
        # Create dumb target
        input_shape = (1, self.decoder.seq_length, self.decoder.input_size)
        db_trg = torch.zeros(input_shape)
        # Forward pass in the decoder
        generated_bar = self.decoder(latent.unsqueeze(0), db_trg, teacher_forcing=False)
        return generated_bar


class Encoder_pianoroll(nn.Module):
    def __init__(self, args, input_dim, hidden_size, latent_size, num_layers):
        """"" This initializes the encoder"""
        super(Encoder_pianoroll, self).__init__()
        self.RNN = nn.LSTM(input_dim, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=True,
                           dropout=0.6)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.latent_size = latent_size

    def forward(self, x, h0, c0):
        batch_size = x.shape[0]
        _, (h, _) = self.RNN(x, (h0, c0))
        h = h.view(self.num_layers, 2, batch_size, -1)
        h = h[-1]
        h = torch.cat([h[0], h[1]], dim=1)
        return h

    def init_hidden(self, args, batch_size=1):
        # Bidirectional lstm so num_layers*2
        return (torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, dtype=torch.float, device=args.device),
                torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, dtype=torch.float, device=args.device))


class Decoder_pianoroll(nn.Module):
    def __init__(self, args, input_size, latent_size, cond_hidden_size, cond_outdim, dec_hidden_size, num_layers,
               num_subsequences, seq_length):
        super(Decoder_pianoroll, self).__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.fc_init_cond = nn.Linear(latent_size, cond_hidden_size * num_layers)
        self.conductor_RNN = nn.LSTM(latent_size // num_subsequences, cond_hidden_size, batch_first=True, num_layers=2,
                                     bidirectional=False, dropout=0.6)
        self.conductor_output = nn.Linear(cond_hidden_size, cond_outdim)
        self.fc_init_dec = nn.Linear(cond_outdim, dec_hidden_size * num_layers)
        self.decoder_RNN = nn.LSTM(cond_outdim + input_size, dec_hidden_size, batch_first=True, num_layers=2,
                                   bidirectional=False, dropout=0.6)
        self.decoder_output = nn.Linear(dec_hidden_size, input_size)
        self.num_subsequences = num_subsequences
        self.input_size = input_size
        self.cond_hidden_size = cond_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.teacher_forcing_ratio = 0.5

    def forward(self, args, latent, target, teacher_forcing):
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
        out = torch.zeros(batch_size, self.seq_length, self.input_size, dtype=torch.float, device=args.device)
        token = torch.zeros(batch_size, subseq_size, self.input_size, dtype=torch.float, device=args.device)
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
