import matplotlib.pyplot as plt
from torch import distributions
import pretty_midi
from data_loaders import data_loader
import random
import numpy as np
import os
import torch
from torch import distributions
from data_loaders.data_loader import maximum


def reconstruction(args, model, epoch, dataset):
    max_global = 128
    # Plot settings
    nrows, ncols = 4, 2  # array of sub-plots
    figsize = np.array([8, 20])  # figure size, inches
    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # generate random index for testing random data
    rand_ind = np.array([random.randint(0, len(dataset)) for i in range(nrows)])
    ind = 0
    for i, axi in enumerate(ax.flat):
        if i % 2 == 0:
            piano_roll = dataset[rand_ind[ind]]
            axi.matshow(piano_roll, alpha=1)
            # write row/col indices as axes' title for identification
            axi.set_title("Original number " + str(rand_ind[ind]))
        else:
            cur_input = dataset[rand_ind[ind]].unsqueeze(0).to(args.device)
            _, _, _, x_reconstruct = model(cur_input)
            x_reconstruct = x_reconstruct.squeeze(0).squeeze(0).detach().cpu()
            axi.matshow(x_reconstruct * 128, alpha=1)
            # write row/col indices as axes' title for identification
            axi.set_title("Reconstruction number " + str(rand_ind[ind]))
            ind += 1

    plt.tight_layout(True)
    # plt.subplots(constrained_layout=True)
    if not os.path.exists(args.figure_reconstruction_path):
        os.makedirs(args.figure_reconstruction_path)
    plt.savefig(args.figure_reconstruction_path + 'epoch_' + str(epoch))
    # plt.show()


def sampling(args, fs=100, program=0):
    latent = distributions.normal.Normal(torch.tensor([0, 0]), torch.tensor([1, 0]))
    generated_bar = args.model().generate(latent)
    notes, frames = generated_bar.shape
    # Generate figure from sampling
    if not os.path.exists(args.sampling_figure):
        os.makedirs(args.sampling_figure)
    for i in range(generated_bar.shape[0]):
        plt.matshow(generated_bar[i].cpu(), alpha=1)
        plt.title("Sampling from latent space")
        plt.tight_layout(True)
        plt.savefig(args.sampling_figure + ".png")
        plt.close()
    # Generate MIDI from sampling
    if not os.path.exists(args.sampling_midi):
        os.makedirs(args.sampling_midi)
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
    # Write out the MIDI data
    print('[Writing MIDI from:]', pm)
    pm.write(args.sampling_midi + ".mid")


def interpolation(model, X, labels, args, a=None, b=None, x_c=None, alpha=0.):
    # Encode samples to the latent space
    z_a, z_b = model.encode(X[labels == a]), model.encode(X[labels == b])
    # Find the centroids of the classes a, b in the latent space
    z_a_centroid = z_a.mean(axis=0)
    z_b_centroid = z_b.mean(axis=0)
    # The interpolation vector pointing from b to a
    z_b2a = z_a_centroid - z_b_centroid
    # Manipulate x_c
    z_c = model.encode(x_c)
    z_c_interp = z_c + alpha * z_b2a
    return model.decode(z_c_interp)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Reconstruction')
#     parser.add_argument('--device', type=str, default='cuda:0', help='device cuda or cpu')
#     parser.add_argument('--midi_path', type=str, default='/fast-1/mathieu/datasets/maestro_folders/train',
#                         help='path to midi folder')
#     parser.add_argument('--model_path', type=str, default='/slow-2/ninon/pyrapro/models_saving/entire_model/',
#                         help='path to the saved model')
#     parser.add_argument('--figure_reconstruction_path', type=str, default='/slow-2/ninon/pyrapro/reconstruction/',
#                         help='path to reconstruction figures')
#     args = parser.parse_args()
#     epoch = 0
#     print("DEBUG BEGIN")
#     reconstruction(args, model, epoch)
#     print("DEBUG END")
