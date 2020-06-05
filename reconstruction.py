import matplotlib.pyplot as plt
import data_loader
from vae import VaeModel
from vae import HierarchicalDecoder, HierarchicalEncoder
import random
import torch
import os

input_dim = 100
enc_hidden_size = 2048
latent_size = 512
cond_hidden_size = 1024
cond_output_dim = 512
dec_hidden_size = 1024
num_layers = 2
num_subsequences = 8
seq_length = 128
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
encoder = HierarchicalEncoder(input_dim=input_dim, enc_hidden_size=enc_hidden_size,
                              latent_size=latent_size)
decoder = HierarchicalDecoder(input_size=input_dim, latent_size=latent_size,
                              cond_hidden_size=cond_hidden_size, cond_outdim=cond_output_dim,
                              dec_hidden_size=dec_hidden_size, num_layers=num_layers,
                              num_subsequences=num_subsequences, seq_length=seq_length)
model = VaeModel(encoder=encoder, decoder=decoder).float().to(device=device)


def reconstruction(midi_path, model_path, figure_saving_path, epoch):
    dataset = data_loader.PianoRollRep(midi_path)
    # Load the entire model
    torch.load(model_path + '_epoch_' + str(epoch) + '.pth')

    # Plot settings
    nrows, ncols = 4, 2  # array of sub-plots
    figsize = [8, 20]  # figure size, inches
    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # generate random index for testing random data
    rand_ind = [random.randint(0, len(dataset)) for i in range(nrows)]
    ind = 0

    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        if i % 2 == 0:
            piano_roll = dataset[rand_ind[ind]]
            axi.matshow(piano_roll, alpha=1)
            # write row/col indices as axes' title for identification
            axi.set_title("Original number " + str(rand_ind[ind]))
        else:
            dataset[rand_ind[ind]][dataset[rand_ind[ind]] > 0] = 1
            mu, sigma, latent, x_reconstruct = model(dataset[rand_ind[ind]].unsqueeze(0))
            x_reconstruct = x_reconstruct.squeeze(0).squeeze(0).detach().numpy()
            axi.matshow(x_reconstruct, alpha=1)
            # write row/col indices as axes' title for identification
            axi.set_title("Reconstruction number " + str(rand_ind[ind]))
            ind += 1

    plt.tight_layout(True)
    # plt.subplots(constrained_layout=True)
    if not os.path.exists(figure_saving_path):
        os.makedirs(figure_saving_path)
    plt.savefig(figure_saving_path + 'epoch_' + str(epoch))
    # plt.show()
