import matplotlib.pyplot as plt
import data_loader
from vae import VaeModel
from vae import HierarchicalDecoder, HierarchicalEncoder
import random
import torch

# Model and data
midi_path = 'midi_short_dataset'
model_path = './models/entire_model/_epoch_2.pth'

dataset = data_loader.PianoRollRep(midi_path)
input_dim = 100
enc_hidden_size = 2048
latent_size = 512
cond_hidden_size = 1024
cond_output_dim = 512
dec_hidden_size = 1024
num_layers = 2
num_subsequences = 8
seq_length = 128
# encoder = m.Encoder(input_dim, enc_hidden_size, latent_size, num_layers)
# decoder = m.Decoder(input_dim, latent_size, cond_hidden_size, cond_output_dim,
#                     dec_hidden_size, num_layers, num_subsequences, seq_length)
# model = torch.nn.Sequential(encoder, decoder)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = HierarchicalEncoder(input_dim=input_dim, enc_hidden_size=enc_hidden_size,
                              latent_size=latent_size)
decoder = HierarchicalDecoder(input_size=input_dim, latent_size=latent_size,
                              cond_hidden_size=cond_hidden_size, cond_outdim=cond_output_dim,
                              dec_hidden_size=dec_hidden_size, num_layers=num_layers,
                              num_subsequences=num_subsequences, seq_length=seq_length)
model = VaeModel(encoder=encoder, decoder=decoder).float().to(device=device)

# Load the state_dict = only weights
# model_state_dict = model.load_state_dict(torch.load(weight_model_path))
# Load the entire model
model_state_dict = torch.load(model_path)
# model.eval()

# Plot settings
nrows, ncols = 6, 2  # array of sub-plots
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
        pianoR = dataset[rand_ind[ind]]
        axi.matshow(pianoR, alpha=1)
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        axi.set_title("Orignal number " + str(rand_ind[ind]))
    else:
        dataset[rand_ind[ind]][dataset[rand_ind[ind]] > 0] = 1
        mu, sigma, latent, x_reconstruct = model(dataset[rand_ind[ind]].unsqueeze(0))
        x_reconstruct = x_reconstruct.squeeze(0).squeeze(0).detach().numpy()
        axi.matshow(x_reconstruct, alpha=1)
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        axi.set_title("Reconstruction number " + str(rand_ind[ind]))
        ind += 1

plt.tight_layout(True)
plt.show()

#
# # generate random index for testing random data
# rand_ind = [random.randint(0,len(dataset)) for i in range(6)]
# ind = 0
#
# for i in range(6):
#     print('-'*50)
#     print(dataset[rand_ind[ind]])
#     print(' ')
#     h_enc, c_enc = model[0].init_hidden(1)
#     mu, log_var = model[0](dataset[rand_ind[ind]].unsqueeze(0), h_enc, c_enc)
#     latent = mu + log_var * torch.randn_like(mu)
#     h_dec, c_dec = model[1].init_hidden(1)
#     x_reconst = model[1](latent, h_dec, c_dec, teacher_forcing=False)
#     print(x_reconst.max(dim=2))
#     ind+=1
