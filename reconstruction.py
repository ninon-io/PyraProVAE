import matplotlib.pyplot as plt
from data_loaders import data_loader
import random
import numpy as np
import os


def reconstruction(args, model, epoch):
    dataset = data_loader.PianoRollRep(args.midi_path)

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
            dataset[rand_ind[ind]][dataset[rand_ind[ind]] > 0] = 1
            cur_input = dataset[rand_ind[ind]].unsqueeze(0).to(args.device)
            _, _, _, x_reconstruct = model(cur_input)
            x_reconstruct = x_reconstruct.squeeze(0).squeeze(0).detach().cpu()
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

