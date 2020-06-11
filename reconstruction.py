import matplotlib.pyplot as plt
import data_loader
from vae import VaeModel
from vae import HierarchicalDecoder, HierarchicalEncoder
import random
import torch
import os


def reconstruction(args, model, epoch):
    dataset = data_loader.PianoRollRep(args.midi_path)
    # Load the entire model
    torch.load(args.model_path + '_epoch_' + str(epoch) + '.pth')

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
            model.to(args.device)
            mu, sigma, latent, x_reconstruct = model(dataset[rand_ind[ind]].unsqueeze(0))
            x_reconstruct = x_reconstruct.squeeze(0).squeeze(0).detach().numpy()
            axi.matshow(x_reconstruct, alpha=1)
            # write row/col indices as axes' title for identification
            axi.set_title("Reconstruction number " + str(rand_ind[ind]))
            ind += 1

    plt.tight_layout(True)
    # plt.subplots(constrained_layout=True)
    if not os.path.exists(args.figure_reconstruction_path):
        os.makedirs(args.figure_reconstruction_path)
    plt.savefig(args.figure_reconstruction_path + 'epoch_' + str(epoch))
    # plt.show()
