import torch
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from guppy import hpy
import os
import pretty_midi
# VAE model
from vae import VaeModel
from vae import HierarchicalDecoder, HierarchicalEncoder

from tensorboardX import SummaryWriter

# Track the memory usage
h = hpy()

# Dimensions of the architecture
input_dim = 100
enc_hidden_size = 2048
latent_size = 512
cond_hidden_size = 1024
cond_output_dim = 512
dec_hidden_size = 1024
num_layers = 2
num_subsequences = 8
seq_length = 128


class Learn:
    def __init__(self, train_loader, test_loader, train_set, test_set, batch_size=512, seed=1, lr=0.01):
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define model
        self.encoder = HierarchicalEncoder(input_dim=input_dim, enc_hidden_size=enc_hidden_size,
                                           latent_size=latent_size)
        self.decoder = HierarchicalDecoder(input_size=input_dim, latent_size=latent_size,
                                           cond_hidden_size=cond_hidden_size, cond_outdim=cond_output_dim,
                                           dec_hidden_size=dec_hidden_size, num_layers=num_layers,
                                           num_subsequences=num_subsequences, seq_length=seq_length)
        self.model = VaeModel(encoder=self.encoder, decoder=self.decoder).float().to(device=self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.iter_train = 1
        self.epoch = 0
        self.iter_test = 0
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

    def train(self, epoch, log_interval=10):
        writer = SummaryWriter('/slow-2/ninon/pyrapro/output/runs')
        print(self.device)
        self.model.train()
        for batch_idx, x in tqdm(enumerate(self.train_loader), total=len(self.train_set) // self.batch_size):
            x = x.to(self.device)
            mu, sigma, latent, x_recon = self.model(x)
            with torch.no_grad():
                log_var = np.log(sigma.detach() ** 2)
            kl_div = - 1 / 2 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            recon_loss = F.mse_loss(x_recon.squeeze(1), x)
            self.recon_loss_mean += recon_loss.detach()
            self.kl_div_mean += kl_div.detach()
            # Training pass
            loss = recon_loss + self.beta * kl_div
            self.loss_mean += loss.detach()
            self.optimizer.zero_grad()
            # Learning with back-propagation
            loss.backward()
            # Optimizes weights
            self.optimizer.step()
            # if batch_idx % log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         self.iter_train, batch_idx * len(x), len(self.train_loader.dataset),
            #         100. * batch_idx / len(self.train_loader), self.loss_mean))
            if self.iter_train > 10 and self.beta < 1:
                self.beta += 0.0025
            self.iter_train += 1
        with torch.no_grad():
            writer.add_scalar('data/loss_mean', self.loss_mean / self.iter_train, epoch)
            writer.add_scalar('data/kl_div_mean', self.kl_div_mean / self.iter_train, epoch)
            writer.add_scalar('data/reconst_loss_mean', self.recon_loss_mean / self.iter_train, epoch)
            writer.close()
        return self.loss_mean, self.kl_div_mean, self.recon_loss_mean

    def test(self, epoch):
        print(self.device)
        writer = SummaryWriter('/slow-2/ninon/pyrapro/output/runs')
        self.model.eval()
        with torch.no_grad():
            for batch_idx, x in tqdm(enumerate(self.test_loader), total=len(self.test_set) // self.batch_size):
                if x.max() != 0:
                    x = x / x.max()
                x = x.to(self.device)
                mu, sigma, latent, x_recon = self.model(x)
                log_var = np.log(sigma ** 2)
                kl_div = - 1 / 2 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                recon_loss = F.mse_loss(x_recon.squeeze(1), x)
                self.recon_loss_mean_test += recon_loss.detach()
                self.kl_div_mean_test += kl_div.detach()
                loss = recon_loss + self.beta * kl_div
                self.loss_mean_test += loss.detach()
                self.iter_test += 1
        with torch.no_grad():
            writer.add_scalar('data/loss_mean_TEST', self.loss_mean_test / self.iter_test, epoch)
            writer.add_scalar('data/kl_div_mean_TEST', self.kl_div_mean_test / self.iter_test, epoch)
            writer.add_scalar('data/reconst_loss_mean_TEST', self.recon_loss_mean_test / self.iter_test, epoch)
        # Print stuffs
        print('\nTest set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            self.loss_mean_test, loss, len(self.test_loader.dataset), 100. * loss / len(self.test_loader)))
        return self.loss_mean_test, self.kl_div_mean_test, self.recon_loss_mean_test

    def save(self, model_weights_saving_path, entire_model_saving_path, epoch):
        # Save entire model
        if not os.path.exists(entire_model_saving_path):
            os.makedirs(entire_model_saving_path)
        torch.save(self.model, entire_model_saving_path + '_epoch_' + str(epoch) + '.pth')
        # Save only the weights
        if not os.path.exists(model_weights_saving_path):
            os.makedirs(model_weights_saving_path)
        torch.save(self.model.state_dict(), model_weights_saving_path + '_epoch_' + str(epoch) + '.pth')

    def resume_training(self, entire_model_saving_path, epoch):  # Specify the wishing epoch resuming here
        model = self.model
        state_dict = torch.load(entire_model_saving_path + '_epoch_' + str(epoch) + '.pth')
        model.eval()

    # TODO: NOT WORKING
    def piano_roll_recon(self, entire_model_saving_path, fs=100, program=0):
        # state_dict = torch.load(entire_model_saving_path)
        # loss_mean, kl_div_mean, recon_loss_mean = learn.train()
        model = self.model
        optimizer = self.optimizer
        checkpoint = torch.load(entire_model_saving_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict((checkpoint['optimizer_state_dict']))
        mu, sigma, latent, x_reconstruct = self.model()
        self.epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        model.eval()
        generated_bar = self.model().generate(latent)
        notes, frames = generated_bar.shape
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=program)
        # Pad 1 column of zeros to acknowledge initial and ending events
        piano_roll = np.pad(generated_bar, [(0, 0), (1, 1)], 'constant')
        # Use changes in velocities to find note on/note off events
        velocity_changes = np.nonzero(np.diff(piano_roll).T)
        # Keep track on velocities and note on times
        prev_velocities = np.zeros(notes, dtype=int)
        note_on_time = np.zeros(notes)

        for time, note in zip(*velocity_changes):
            # Use time + 1 because of padding above
            velocity = piano_roll[notes, time + 1]
            time = time / fs
            if velocity > 0:
                if prev_velocities[note] == 0:
                    note_on_time[note] = time
                    prev_velocities[note] = velocity
            else:
                pm_note = pretty_midi.Note(
                    velocity=prev_velocities[note],
                    pitch=note,
                    start=note_on_time[note],
                    end=time)
                instrument.notes.append(pm_note)
                prev_velocities[note]=0
            pm.instruments.append(instrument)
            return pm
        print('PianoRoll', pm)

