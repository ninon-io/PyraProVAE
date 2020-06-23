import torch
from torch import distributions
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from guppy import hpy
import os
import pretty_midi

from tensorboardX import SummaryWriter

# Track the memory usage
h = hpy()


class Learn:
    def __init__(self, args, train_loader, validate_loader, test_loader, train_set, validate_set, test_set):
        self.iter_train = 1
        self.epoch = torch.zeros(1).to(args.device)
        self.iter_test = torch.zeros(1).to(args.device)
        self.train_loader = train_loader
        self.validate_loader = validate_loader
        self.test_loader = test_loader
        self.train_set = train_set
        self.validate_set = validate_set
        self.test_set = test_set
        # Settings
        self.loss_mean = torch.zeros(1).to(args.device)
        self.recon_loss_mean = torch.zeros(1).to(args.device)
        self.kl_div_mean = torch.zeros(1).to(args.device)
        self.beta = torch.zeros(1).to(args.device)

        self.loss_mean_validate = torch.zeros(1).to(args.device)
        self.recon_loss_mean_validate = torch.zeros(1).to(args.device)
        self.kl_div_mean_validate = torch.zeros(1).to(args.device)
        self.beta_validate = torch.zeros(1).to(args.device)

        self.loss_mean_test = torch.zeros(1).to(args.device)
        self.kl_div_mean_test = torch.zeros(1).to(args.device)
        self.recon_loss_mean_test = torch.zeros(1).to(args.device)

    def train(self, model, optimizer, args, epoch):
        writer = SummaryWriter('/slow-2/ninon/pyrapro/output/runs')
        print('train pass on:', args.device)
        model.train()
        for batch_idx, x in tqdm(enumerate(self.train_loader), total=len(self.train_set) // args.batch_size):
            x = x.to(args.device, non_blocking=True)
            mu, sigma, latent, x_recon = model(x)
            print("Beautiful batch size is %d" % (x.shape[0]))
            print("Are you a mofo silence ?")
            print(torch.sum(x, (1, 2)))
            #for i in range(x.shape[0]):
            #    import matplotlib.pyplot as plt
            #    plt.matshow(x[i].cpu(), alpha=1)
            #    plt.savefig("reconstruction/batch_%d_example_%d.png" % (batch_idx, i))
            #    plt.close()
            print("Cheese nan dans mu ? - %d" % (torch.sum(torch.isnan(mu))))
            print("Cheese nan dans sigma ? - %d" % (torch.sum(torch.isnan(sigma))))
            print("Cheese nan dans latent ? - %d" % (torch.sum(torch.isnan(latent))))
            print("Cheese nan dans x_recon ? - %d" % (torch.sum(torch.isnan(x_recon))))
            #with torch.no_grad():
            log_var = torch.log(sigma ** 2)
            kl_div = - 1 / 2 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            recon_loss = F.mse_loss(x_recon.squeeze(1), x)
            #self.recon_loss_mean += recon_loss.detach()
            #self.kl_div_mean += kl_div.detach()
            # Training pass
            loss = recon_loss + self.beta * kl_div
            print("KL Loss is : %f" % (kl_div.item()))
            print("Recons loss is : %f" % (recon_loss.item()))
            print("Loss is : %f" % (loss.item()))
            #self.loss_mean += loss.detach()
            optimizer.zero_grad()
            # Learning with back-propagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            loss.backward()
            # Optimizes weights
            optimizer.step()
        if self.iter_train > 10 and self.beta < 1:
            self.beta += 0.0025
        self.iter_train += 1
        with torch.no_grad():
            writer.add_scalar('data/loss_mean', self.loss_mean / self.iter_train, epoch)
            writer.add_scalar('data/kl_div_mean', self.kl_div_mean / self.iter_train, epoch)
            writer.add_scalar('data/reconst_loss_mean', self.recon_loss_mean / self.iter_train, epoch)
            writer.close()
        return self.loss_mean, self.kl_div_mean, self.recon_loss_mean

    def validate(self, model, args, epoch):
        writer = SummaryWriter('/slow-2/ninon/pyrapro/output/runs')
        print('validation pass on:', args.device)  # print(f"validation pass on: {args.device}")
        model.eval()
        with torch.no_grad():
            for batch_idx, x in tqdm(enumerate(self.validate_loader), total=len(self.validate_set) // args.batch_size):
                x = x.to(args.device)
                mu, sigma, latent, x_recon = model(x)
                print("Cheese nan dans mu ? - %d" % (torch.sum(torch.isnan(mu))))
                print("Cheese nan dans sigma ? - %d" % (torch.sum(torch.isnan(sigma))))
                print("Cheese nan dans latent ? - %d" % (torch.sum(torch.isnan(latent))))
                print("Cheese nan dans x_recon ? - %d" % (torch.sum(torch.isnan(x_recon))))
                log_var = torch.log(sigma.detach() ** 2)
                kl_div = - 1 / 2 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                recon_loss = F.mse_loss(x_recon.squeeze(1), x)
                self.recon_loss_mean_validate += recon_loss.detach()
                self.kl_div_mean_validate += kl_div.detach()
                loss = recon_loss + self.beta * kl_div
                self.loss_mean_validate += loss.detach()
        with torch.no_grad():
            writer.add_scalar('data/loss_mean_VALID', self.loss_mean_validate / self.iter_train, epoch)
            writer.add_scalar('data/kl_div_mean_VALID', self.kl_div_mean_validate / self.iter_train, epoch)
            writer.add_scalar('data/reconst_loss_mean_VALID', self.recon_loss_mean_validate / self.iter_train, epoch)
            writer.close()
        return self.loss_mean, self.kl_div_mean, self.recon_loss_mean

    def test(self, model, args, epoch):
        print('test pass:', args.device)
        writer = SummaryWriter('/slow-2/ninon/pyrapro/output/runs')
        model.eval()
        with torch.no_grad():
            for batch_idx, x in tqdm(enumerate(self.test_loader), total=len(self.test_set) // args.batch_size):
                if x.max() != 0:
                    x = x / x.max()
                x = x.to(args.device)
                mu, sigma, latent, x_recon = model(x)
                print("Cheese nan dans mu ? - %d" % (torch.sum(torch.isnan(mu))))
                print("Cheese nan dans sigma ? - %d" % (torch.sum(torch.isnan(sigma))))
                print("Cheese nan dans latent ? - %d" % (torch.sum(torch.isnan(latent))))
                print("Cheese nan dans x_recon ? - %d" % (torch.sum(torch.isnan(x_recon))))
                log_var = torch.log(sigma ** 2)
                kl_div = - 1 / 2 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                recon_loss = F.mse_loss(x_recon.squeeze(1), x)
                self.recon_loss_mean_test += recon_loss.detach()
                self.kl_div_mean_test += kl_div.detach()
                loss = recon_loss + self.beta * kl_div
                self.loss_mean_test += loss.detach()
        with torch.no_grad():
            writer.add_scalar('data/loss_mean_TEST', self.loss_mean_test / self.iter_test, epoch)
            writer.add_scalar('data/kl_div_mean_TEST', self.kl_div_mean_test / self.iter_test, epoch)
            writer.add_scalar('data/reconst_loss_mean_TEST', self.recon_loss_mean_test / self.iter_test, epoch)
        return self.loss_mean_test, self.kl_div_mean_test, self.recon_loss_mean_test

    def save(self, model, args, epoch):
        # Save entire model
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
        torch.save(model, args.model_path + '_epoch_' + str(epoch) + '.pth')
        # Save only the weights
        if not os.path.exists(args.weights_path):
            os.makedirs(args.weights_path)
        torch.save(model.state_dict(), args.weights_path + '_epoch_' + str(epoch) + '.pth')

    def resume_training(self, args, model, epoch):  # Specify the wishing epoch resuming here
        torch.load(args.model_path + '_epoch_' + str(epoch) + '.pth')
        model.eval()

    # TODO: dynamic code depending of latent size
    def piano_roll_recon(self, model, entire_model_saving_path, fs=100, program=0):
        latent = distributions.normal.Normal(torch.tensor([0, 0]), torch.tensor([1, 0]))
        generated_bar = model().generate(latent)
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
                prev_velocities[note] = 0
            pm.instruments.append(instrument)
            return pm
        print('PianoRoll', pm)
