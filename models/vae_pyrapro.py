import torch
from torch import nn
import random


class VaeModel(nn.Module):
    def __init__(self, encoder, decoder, teacher_forcing=True):
        super(VaeModel, self).__init__()
        self.tf = teacher_forcing
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_to_mu = nn.Linear(2 * encoder.enc_hidden_size, encoder.latent_size)
        self.hidden_to_sigma = nn.Linear(2 * encoder.enc_hidden_size, encoder.latent_size)

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
        x_reconstruct = self.decoder(latent, x, teacher_forcing=False)

        return mu, sigma, latent, x_reconstruct

    # Generate bar from latent space
    def generate(self, latent):
        # Create dumb target
        input_shape = (1, self.decoder.seq_length, self.decoder.input_size)
        trg = torch.zeros(input_shape)
        # Forward pass in the decoder
        generated_bar = self.decoder(latent.unsqueeze(0), trg, teacher_forcing=False)

        return generated_bar


class HierarchicalEncoder(nn.Module):
    def __init__(self, input_dim, enc_hidden_size, latent_size, num_layers=2):
        super(HierarchicalEncoder, self).__init__()
        self.enc_hidden_size = enc_hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.RNN = nn.LSTM(input_dim, enc_hidden_size, batch_first=True, num_layers=num_layers,
                           bidirectional=True, dropout=0.6)

    def init_hidden(self, batch_size=1):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # initialize the the hidden state // Bidirectionnal so num_layers * 2 \\
        return (torch.zeros(self.num_layers * 2, batch_size, self.enc_hidden_size, dtype=torch.float, device=device),
                torch.zeros(self.num_layers * 2, batch_size, self.enc_hidden_size, dtype=torch.float, device=device))

    def forward(self, x, h0, c0):
        batch_size = x.shape[0]
        _, (h, _) = self.RNN(x.float(), (h0.float(), c0.float()))  # TODO: UNIQUE DIFFERENCE
        h = h.view(self.num_layers, 2, batch_size, -1)
        h = h[-1]
        h = torch.cat([h[0], h[1]], dim=1)
        return h


class HierarchicalDecoder(nn.Module):  # TODO: Put batch norm + ReLU
    def __init__(self, input_size, latent_size, cond_hidden_size, cond_outdim, dec_hidden_size, num_layers,
                 num_subsequences, seq_length):
        super(HierarchicalDecoder, self).__init__()
        self.num_subsequences = num_subsequences
        self.input_size = input_size
        self.cond_hidden_size = cond_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.teacher_forcing_ratio = 0  # TFR must be 0: worsen the training except for non piano-roll representation

        # Define init for architecture: first conductor then decoder
        self.tanh = nn.Tanh()
        # TODO: use one of them eventually?
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.fc_init_cond = nn.Linear(latent_size, cond_hidden_size * num_layers)
        self.conductor_RNN = nn.LSTM(latent_size // num_subsequences, cond_hidden_size, batch_first=True, num_layers=2,
                                     bidirectional=False, dropout=0.6)
        self.conductor_output = nn.Linear(cond_hidden_size, cond_outdim)
        self.fc_init_dec = nn.Linear(cond_outdim, dec_hidden_size * num_layers)
        self.decoder_RNN = nn.LSTM(cond_outdim + input_size, dec_hidden_size, batch_first=True, num_layers=2,
                                   bidirectional=False, dropout=0.6)
        self.decoder_output = nn.Linear(dec_hidden_size, input_size)

    def forward(self, latent, target, teacher_forcing):
        batch_size = latent.shape[0]
        subseq_size = self.seq_length // self.num_subsequences
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("[RNN Style] = Cheese nan dans latent ? - %d" % (torch.sum(torch.isnan(latent))))
        print("[RNN Style] = Cheese nan dans fc_init_cond ? - %d" % (torch.sum(torch.isnan(self.fc_init_cond(latent)))))

        # Get the initial states of the conductor
        h0_cond = self.tanh(self.fc_init_cond(latent)).view(self.num_layers, batch_size, -1).contiguous()
        print("[RNN Style] = Cheese nan dans h0_cond ? - %d" % (torch.sum(torch.isnan(h0_cond))))
        # Divide the latent space into subsequences:
        latent = latent.view(batch_size, self.num_subsequences, -1)
        # Pass through the conductor
        h0_ninja = torch.zeros(self.num_layers, batch_size, self.cond_hidden_size, dtype=torch.float, device=device)
        c0_ninja = torch.zeros(self.num_layers, batch_size, self.cond_hidden_size, dtype=torch.float, device=device)

        # Taking latent (batch x latent_dim)
        subseq_embeddings, _ = self.conductor_RNN(latent, (h0_ninja, c0_ninja))
        print("[RNN Style] = Cheese nan dans subseq_embeddings ? - %d" % (torch.sum(torch.isnan(subseq_embeddings))))
        # Output is (batch x time x rnn_dim)
        subseq_embeddings = self.conductor_output(subseq_embeddings)
        print("[RNN Style] = Cheese nan dans latent ? - %d" % (torch.sum(torch.isnan(subseq_embeddings))))

        # Get the initial state of decoder
        h0s_dec = self.tanh(self.fc_init_dec(subseq_embeddings)).view(self.num_layers, batch_size,
                                                                      self.num_subsequences, -1).contiguous()
        print("[RNN Style] = Cheese nan dans latent ? - %d" % (torch.sum(torch.isnan(h0s_dec))))

        # Init the output seq and the first token to 0 tensors
        out = torch.zeros(batch_size, self.seq_length, self.input_size, dtype=torch.float, device=device)
        token = torch.zeros(batch_size, subseq_size, self.input_size, dtype=torch.float, device=device)

        # autoregressively output token
        for subseq in range(self.num_subsequences):
            subseq_embedding = subseq_embeddings[:, subseq, :]
            h0_dec = h0s_dec[:, :, subseq, :].contiguous()
            c0_dec = h0s_dec[:, :, subseq, :].contiguous()
            # Concat the previous token and the current subseq embedding as input
            dec_input = torch.cat((token, subseq_embedding.unsqueeze(1).expand(-1, subseq_size, -1)), -1)
            # Pass through the decoder
            token, (h0_dec, c0_dec) = self.decoder_RNN(dec_input, (h0_dec, c0_dec))
            token = self.decoder_output(token)
            # Fill the out tensor with the token
            out[:, subseq * subseq_size:((subseq + 1) * subseq_size), :] = token
            # If teacher forcing replace the output token by the real one sometimes
            if teacher_forcing:
                if random.random() <= self.teacher_forcing_ratio:
                    token = target[:, subseq * subseq_size: ((subseq + 1) * subseq_size), :]
        print("[RNN Style] = Cheese nan dans latent ? - %d" % (torch.sum(torch.isnan(out))))
        return out


class Decoder(nn.Module):

    def __init__(self, input_size, latent_size, cond_hidden_size, cond_outdim, hidden_size, num_layers, num_subsequences, seq_length):
        super(Decoder, self).__init__()
        #self.latent_to_conductor = nn.Linear(latent_size, latent_size)
        self.tanh = nn.Tanh()
        self.conductor_RNN = nn.LSTM(latent_size, cond_hidden_size, batch_first=True, num_layers=2, bidirectional=False)
        self.conductor_output = nn.Linear(cond_hidden_size, cond_outdim)
        self.decoder_RNN = nn.LSTM(cond_outdim + input_size, hidden_size, batch_first=True, num_layers=2, bidirectional=False)
        self.decoder_output = nn.Linear(hidden_size, input_size)
        self.softmax = nn.Softmax()
        self.num_subsequences = num_subsequences
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.teacher_forcing_ratio = 0.5

    def forward(self, latent, target, teacher_forcing):
        batch_size = latent.shape[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target = torch.nn.functional.one_hot(target.long(), 389).float()
        h0, c0 = self.init_hidden(batch_size)
        out = torch.zeros(batch_size, self.seq_length, self.input_size, dtype=torch.float, device=device)
        prev_note = torch.zeros(batch_size, 1, self.input_size, dtype=torch.float, device=device)
        for subseq_idx in range(self.num_subsequences):
            subseq_embedding, (h0, c0) = self.conductor_RNN(latent.unsqueeze(1), (h0, c0))
            subseq_embedding = self.tanh(self.conductor_output(subseq_embedding))
            # Initialize lower decoder hidden state
            h0_dec = (torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device),
                      torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device))

            use_teacher_forcing = False #if random.random() < self.teacher_forcing_ratio else False
            for note_idx in range(int(self.seq_length/self.num_subsequences)):
                e = torch.cat((prev_note, subseq_embedding), -1)
                prev_note, h0_dec = self.decoder_RNN(e, h0_dec)
                prev_note = self.tanh(self.decoder_output(prev_note))

                idx = subseq_idx * self.seq_length/self.num_subsequences + note_idx
                out[:, int(idx), :] = prev_note.squeeze()
                if use_teacher_forcing :
                    prev_note = target[:,int(idx),:].unsqueeze(1)
        return out

    def init_hidden(self, batch_size=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device))