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
import argparse
from models.vae_gru import VAEKawai


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
            x_reconstruct, _, _ = model(cur_input)
            x_reconstruct = x_reconstruct[0].detach().cpu()
            if args.num_classes > 1:
                x_reconstruct = torch.argmax(x_reconstruct, dim=0)
            axi.matshow(x_reconstruct, alpha=1)
            # write row/col indices as axes' title for identification
            axi.set_title("Reconstruction number " + str(rand_ind[ind]))
            ind += 1

    plt.tight_layout(True)
    if not os.path.exists(args.figure_reconstruction_path):
        os.makedirs(args.figure_reconstruction_path)
    plt.savefig(args.figure_reconstruction_path + 'epoch_' + str(epoch))
    # plt.show()


def sampling(args, fs=100, program=0):
    # Create normal distribution representing latent space
    latent = distributions.normal.Normal(torch.tensor([0, 0], dtype=torch.float), torch.tensor([1, 0], dtype=torch.float))
    # Sampling random from latent space
    z = latent.sample(sample_shape=torch.Size([64, args.latent_size]))
    z = z.view([128, 512])
    # Pass through the decoder
    z = z.transpose(0, 1)
    generated_bar = model.decoder(z)
    # Generate figure from sampling
    if not os.path.exists(args.sampling_figure):
        os.makedirs(args.sampling_figure)
    generated_bar = generated_bar.detach().cpu()
    if args.num_classes > 1:
        generated_bar = torch.argmax(generated_bar, dim=0)
    plt.matshow(generated_bar, alpha=1)
    plt.title("Sampling")
    plt.savefig(args.sampling_figure + 'sampling.png')

    # Generate MIDI from sampling
    # if not os.path.exists(args.sampling_midi):
    #     os.makedirs(args.sampling_midi)
    # pm = pretty_midi.PrettyMIDI()
    # print('bar:', generated_bar.shape)
    # # generated_bar = torch.mean(generated_bar, dim=2)
    # print('bar_gen_mean:', generated_bar.shape)
    # notes, frames = generated_bar.shape
    # instrument = pretty_midi.Instrument(program=program)
    # # Pad 1 column of zeros to acknowledge initial and ending events
    # piano_roll = np.pad(generated_bar.detach(), [(0, 0), (1, 1)], 'constant')
    # # Use changes in velocities to find note on/note off events
    # velocity_changes = np.nonzero(np.diff(piano_roll).T)
    # # Keep track on velocities and note on times
    # prev_velocities = np.zeros(notes, dtype=int)
    # note_on_time = np.zeros(notes)
    #
    # for time, note in zip(*velocity_changes):
    #     # Use time + 1 because of padding above
    #     velocity = piano_roll[notes, time + 1]
    #     time = time / fs
    #     if velocity > 0:
    #         if prev_velocities[note] == 0:
    #             note_on_time[note] = time
    #             prev_velocities[note] = velocity
    #     else:
    #         pm_note = pretty_midi.Note(
    #             velocity=prev_velocities[note],
    #             pitch=note,
    #             start=note_on_time[note],
    #             end=time)
    #         instrument.notes.append(pm_note)
    #         prev_velocities[note] = 0
    #     pm.instruments.append(instrument)
    #     return pm
    # # Write out the MIDI data
    # print('[Writing MIDI from:]', pm)
    # pm.write(args.sampling_midi + ".mid")


def interpolation(x, labels, args, a=None, b=None, x_c=None):
    x_a, x_b =
    # Encode samples to the latent space
    z_a, z_b = model.encode(x[labels == a]), model.encode(x[labels == b])
    # Find the centroids of the classes a, b in the latent space
    z_a_centroid = z_a.mean(axis=0)
    z_b_centroid = z_b.mean(axis=0)
    # The interpolation vector pointing from b to a
    z_b2a = z_a_centroid - z_b_centroid
    # Manipulate x_c
    z_c = model.encode(x_c)
    # Run through alpha values
    interp = []
    alpha_values = np.linspace(0, 1, args.n_steps)
    for alpha in alpha_values:
        z_c_interp = z_c + alpha * z_b2a
        interp.append(model.decode(z_c_interp))
    # Draw interpolation
    for v in interp:
        for i in range(v.shape[0]):
            plt.matshow(interp[i].cpu(), alpha=1)
            plt.title("Interpolation")
            plt.tight_layout(True)
            plt.savefig(args.interp_figure + "interpolation.png")
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyraProVAE')
    # Device Information
    parser.add_argument('--device', type=str, default='cuda:0', help='device cuda or cpu')
    # Data Parameters
    parser.add_argument('--midi_path', type=str, default='/Squarp/PyraProVAE/midi_short_dataset', help='path to midi folder')
    parser.add_argument("--test_size", type=float, default=0.2, help="% of data used in test set")
    parser.add_argument("--valid_size", type=float, default=0.2, help="% of data used in valid set")
    parser.add_argument("--dataset", type=str, default="nottingham",
                        help="maestro | nottingham | bach_chorales | midi_folder")
    parser.add_argument("--shuffle_data_set", type=int, default=1, help='')
    # Novel arguments
    parser.add_argument('--frame_bar', type=int, default=64, help='put a power of 2 here')
    parser.add_argument('--score_type', type=str, default='mono', help='use mono measures or poly ones')
    parser.add_argument('--score_sig', type=str, default='4_4', help='rhythmic signature to use (use "all" to bypass)')
    # parser.add_argument('--data_keys',      type=str, default='C',      help='transpose all tracks to a given key')
    parser.add_argument('--data_normalize', type=int, default=1, help='normalize the data')
    parser.add_argument('--data_binarize', type=int, default=1, help='binarize the data')
    parser.add_argument('--data_pitch', type=int, default=1, help='constrain pitches in the data')
    parser.add_argument('--data_export', type=int, default=0, help='recompute the dataset (for debug purposes)')
    parser.add_argument('--data_augment', type=int, default=1, help='use data augmentation')
    # Model Saving and reconstruction
    parser.add_argument('--model_path', type=str, default='/home/ninon/Squarp/PyraProVAE/saving_model/',
                        help='path to the saved model')
    parser.add_argument('--tensorboard_path', type=str, default='output/', help='path to the saved model')
    parser.add_argument('--weights_path', type=str, default='/home/ninon/Squarp/PyraProVAE/models_saving/weights/',
                        help='path to the saved model')
    parser.add_argument('--figure_reconstruction_path', type=str,
                        default='/home/ninon/Squarp/PyraProVAE/reconstruction/',
                        help='path to reconstruction figures')
    parser.add_argument('--sampling_midi', type=str, default='/home/ninon/Squarp/PyraProVAE/sampling/midi/',
                        help='path to MIDI reconstruction from sampling')
    parser.add_argument('--sampling_figure', type=str, default='/home/ninon/Squarp/PyraProVAE/sampling/figure/',
                        help='path to visual reconstruction from sampling')
    # Model Parameters
    parser.add_argument("--model", type=str, default="vae_kawai", help='PyraPro | vae_mathieu | ae')
    # PyraPro and vae_mathieu specific parameters: dimensions of the architecture
    parser.add_argument('--enc_hidden_size', type=int, default=2048, help='do not touch if you do not know')
    parser.add_argument('--latent_size', type=int, default=512, help='do not touch if you do not know')
    parser.add_argument('--cond_hidden_size', type=int, default=1024, help='do not touch if you do not know')
    parser.add_argument('--cond_output_dim', type=int, default=512, help='do not touch if you do not know')
    parser.add_argument('--dec_hidden_size', type=int, default=1024, help='do not touch if you do not know')
    parser.add_argument('--num_layers', type=int, default=2, help='do not touch if you do not know')
    parser.add_argument('--num_subsequences', type=int, default=8, help='do not touch if you do not know')
    parser.add_argument('--num_classes', type=int, default=2, help='number of velocity classes')
    # Optimization parameters
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--subsample', type=int, default=0, help='train on subset')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--nbworkers', type=int, default=3, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    # Interpolation parameters
    parser.add_argument('--nb_steps', type=int, default=10, help='nb steps for the interpolation')
    # parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    # parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    # Parse the arguments
    args = parser.parse_args()
    print("[DEBUG BEGIN]")
    epoch = 260
    model = torch.load(args.model_path + '_epoch_' + str(epoch) + '.pth', map_location=torch.device('cpu'))
    sampling(args)
    # interpolation(model, )
    print("[DEBUG END]")
