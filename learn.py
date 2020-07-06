import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import os

from tensorboardX import SummaryWriter


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

    def train(self, model, optimizer, criterion, args, epoch):
        writer = SummaryWriter(args.tensorboard_path)
        print(f"train pass on: {args.device}")
        model.train()
        self.loss_mean = torch.zeros(1).to(args.device)
        self.recon_loss_mean = torch.zeros(1).to(args.device)
        self.kl_div_mean = torch.zeros(1).to(args.device)
        for batch_idx, x in tqdm(enumerate(self.train_loader), total=len(self.train_set) // args.batch_size):
            x = x.to(args.device, non_blocking=True)
            mu, sigma, latent, x_recon = model(x)
            log_var = sigma
            kl_div = - 1 / 2 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp().sqrt())
            recon_loss = criterion(x_recon, x)
            self.recon_loss_mean += recon_loss.detach()
            self.kl_div_mean += kl_div.detach()
            # Training pass
            loss = recon_loss + self.beta * kl_div
            self.loss_mean += loss.detach()
            optimizer.zero_grad()
            # Learning with back-propagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.75)
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
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

    def validate(self, model, criterion, args, epoch):
        writer = SummaryWriter(args.tensorboard_path)
        print(f"validation pass on: {args.device}")
        model.eval()
        with torch.no_grad():
            for batch_idx, x in tqdm(enumerate(self.validate_loader), total=len(self.validate_set) // args.batch_size):
                x = x.to(args.device)
                mu, sigma, latent, x_recon = model(x)
                log_var = sigma
                kl_div = - 1 / 2 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                recon_loss = criterion(x_recon, x)
                self.recon_loss_mean_validate += recon_loss.detach()
                self.kl_div_mean_validate += kl_div.detach()
                loss = recon_loss + self.beta * kl_div
                self.loss_mean_validate += loss.detach()
        with torch.no_grad():
            writer.add_scalar('data/loss_mean_VALID', self.loss_mean_validate, epoch)
            writer.add_scalar('data/kl_div_mean_VALID', self.kl_div_mean_validate, epoch)
            writer.add_scalar('data/reconst_loss_mean_VALID', self.recon_loss_mean_validate, epoch)
            writer.close()
        return self.loss_mean_validate, self.kl_div_mean_validate, self.recon_loss_mean_validate

    def test(self, model, criterion, args, epoch):
        writer = SummaryWriter(args.tensorboard_path)
        print(f"test pass on: {args.device}")
        model.eval()
        with torch.no_grad():
            for batch_idx, x in tqdm(enumerate(self.test_loader), total=len(self.test_set) // args.batch_size):
                x = x.to(args.device)
                mu, sigma, latent, x_recon = model(x)
                log_var = sigma
                kl_div = - 1 / 2 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                recon_loss = criterion(x_recon, x)
                self.recon_loss_mean_test += recon_loss.detach()
                self.kl_div_mean_test += kl_div.detach()
                loss = recon_loss + self.beta * kl_div
                self.loss_mean_test += loss.detach()
        with torch.no_grad():
            writer.add_scalar('data/loss_mean_TEST', self.loss_mean_test, epoch)
            writer.add_scalar('data/kl_div_mean_TEST', self.kl_div_mean_test, epoch)
            writer.add_scalar('data/reconst_loss_mean_TEST', self.recon_loss_mean_test, epoch)
            writer.close()
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

