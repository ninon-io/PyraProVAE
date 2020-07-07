import torch
from torch import nn
import random


class VaeModel(nn.Module):
    def __init__(self, encoder, decoder, args, teacher_forcing=True):
        super(VaeModel, self).__init__()
        self.tf = teacher_forcing
        self.device = args.device
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_to_mu = nn.Linear(2 * encoder.enc_hidden_size, encoder.latent_size)
        self.hidden_to_sigma = nn.Linear(2 * encoder.enc_hidden_size, encoder.latent_size)

    def forward(self, x):
        # Encoder pass
        m = nn.ReLU()
        batch_size = m(x).size(0)
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


class HierarchicalEncoder(nn.Module):
    def __init__(self, args):
        super(HierarchicalEncoder, self).__init__()
        self.enc_hidden_size = args.enc_hidden_size
        self.latent_size = args.latent_size
        self.num_layers = args.num_layers
        self.device = args.device

        # Define the LSTM layer
        self.RNN = nn.LSTM(args.input_size[0], args.enc_hidden_size, batch_first=True, num_layers=args.num_layers,
                           bidirectional=True, dropout=0.6)

    def init_hidden(self, batch_size=1):
        # initialize the the hidden state // Bidirectionnal so num_layers * 2 \\
        return (torch.zeros(self.num_layers * 2, batch_size, self.enc_hidden_size, dtype=torch.float, device=self.device),
                torch.zeros(self.num_layers * 2, batch_size, self.enc_hidden_size, dtype=torch.float, device=self.device))

    def forward(self, x, h0, c0):
        batch_size = x.shape[0]
        x = x.transpose(1, 2)
        _, (h, _) = self.RNN(x, (h0, c0))
        h = h.view(self.num_layers, 2, batch_size, -1)
        h = h[-1]
        h = torch.cat([h[0], h[1]], dim=1)
        return h


class HierarchicalDecoder(nn.Module):
    def __init__(self, args):
        super(HierarchicalDecoder, self).__init__()
        self.device = args.device
        self.num_subsequences = args.num_subsequences
        self.input_size = args.input_size[0]
        self.cond_hidden_size = args.cond_hidden_size
        self.dec_hidden_size = args.dec_hidden_size
        self.num_layers = args.num_layers
        self.seq_length = args.input_size[1]
        self.subseq_size = self.seq_length // self.num_subsequences
        self.teacher_forcing_ratio = 0.5
        self.num_classes = args.num_classes
        self.tanh = nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = nn.ReLU()
        self.fc_init_cond = nn.Linear(args.latent_size, args.cond_hidden_size * args.num_layers)
        self.conductor_RNN = nn.LSTM(args.latent_size // args.num_subsequences, args.cond_hidden_size, batch_first=True,
                                     num_layers=2,
                                     bidirectional=False, dropout=0.6)
        self.conductor_output = nn.Linear(args.cond_hidden_size, args.cond_output_dim)
        self.fc_init_dec = nn.Linear(args.cond_output_dim, args.dec_hidden_size * args.num_layers)
        self.decoder_RNN = nn.LSTM(args.cond_output_dim + (self.input_size * self.num_classes), args.dec_hidden_size, batch_first=True,
                                   num_layers=2,
                                   bidirectional=False, dropout=0.6)
        self.decoder_output = nn.Linear(args.dec_hidden_size, self.input_size * self.num_classes)

    def forward(self, latent, target, teacher_forcing):
        batch_size = latent.shape[0]
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
        out = torch.zeros(batch_size, self.seq_length, self.input_size * self.num_classes, dtype=torch.float, device=self.device)
        token = torch.zeros(batch_size, self.subseq_size, self.input_size * self.num_classes, dtype=torch.float, device=self.device)
        # autoregressivly output tokens
        for sub in range(self.num_subsequences):
            subseq_embedding = subseq_embeddings[:, sub, :]
            h0_dec = h0s_dec[:, :, sub, :].contiguous()
            c0_dec = h0s_dec[:, :, sub, :].contiguous()
            # Concat the previous token and the current sub embedding as input
            dec_input = torch.cat((token, subseq_embedding.unsqueeze(1).expand(-1, self.subseq_size, -1)), -1)
            # Pass through the decoder
            token, (h0_dec, c0_dec) = self.decoder_RNN(dec_input, (h0_dec, c0_dec))
            token = self.decoder_output(token)
            token = self.relu(token)
            # Fill the out tensor with the token
            out[:, sub * self.subseq_size:((sub + 1) * self.subseq_size), :] = token
            # If teacher_forcing replace the output token by the real one sometimes
            if teacher_forcing:
                if random.random() <= self.teacher_forcing_ratio:
                    token = target[:, :, sub * self.subseq_size:((sub + 1) * self.subseq_size)].transpose(1, 2)
        out = out.transpose(1, 2)
        out = out.view(batch_size, self.num_classes, self.input_size, -1)
        return out


class Decoder(nn.Module):  # from Mathieu, simplified decoder

    def __init__(self, args):
        super(Decoder, self).__init__()
        #self.latent_to_conductor = nn.Linear(latent_size, latent_size)
        self.device = args.device
        self.tanh = nn.Tanh()
        self.conductor_RNN = nn.LSTM(args.latent_size, args.cond_hidden_size, batch_first=True, num_layers=2, bidirectional=False)
        self.conductor_output = nn.Linear(args.cond_hidden_size, args.cond_outdim)
        self.decoder_RNN = nn.LSTM(args.cond_outdim + args.input_size, args.hidden_size, batch_first=True, num_layers=2, bidirectional=False)
        self.decoder_output = nn.Linear(args.hidden_size, args.input_size)
        self.softmax = nn.Softmax()
        self.num_subsequences = args.num_subsequences
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.seq_length = args.seq_length
        self.teacher_forcing_ratio = 0.5

    def forward(self, latent, target, teacher_forcing, args):
        batch_size = latent.shape[0]
        target = torch.nn.functional.one_hot(target.long(), 389).float()
        h0, c0 = self.init_hidden(batch_size)
        out = torch.zeros(batch_size, self.seq_length, self.input_size, dtype=torch.float, device=args.device)
        prev_note = torch.zeros(batch_size, 1, self.input_size, dtype=torch.float, device=args.device)
        for subseq_idx in range(self.num_subsequences):
            subseq_embedding, (h0, c0) = self.conductor_RNN(latent.unsqueeze(1), (h0, c0))
            subseq_embedding = self.tanh(self.conductor_output(subseq_embedding))
            # Initialize lower decoder hidden state
            h0_dec = (torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=args.device),
                      torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=args.device))

            use_teacher_forcing = False  # if random.random() < self.teacher_forcing_ratio else False
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
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=self.device))
