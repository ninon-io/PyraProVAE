import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from texttable import Texttable
import seaborn as sns
import pandas

# Beautify the plots
large = 26; med = 18; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (8, 5),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': large,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
#%matplotlib inline

# Argument Parser
parser = argparse.ArgumentParser()
# Device Information
parser.add_argument('--device', type=str, default='cuda:0', help='device cuda or cpu')
# Data information
parser.add_argument('--midi_path', type=str, default='/fast-1/mathieu/datasets/', help='path to midi folder')
parser.add_argument("--dataset", type=str, default="nottingham", help="maestro | nottingham | bach_chorales | midi_folder")
parser.add_argument('--score_type',     type=str, default='mono',       help='use mono measures or poly ones')
parser.add_argument('--score_sig',      type=str, default='4_4',        help='rhythmic signature to use (use "all" to bypass)')
parser.add_argument('--data_binarize',  type=int, default=1,            help='binarize the data')
parser.add_argument('--data_augment',   type=int, default=1,            help='use data augmentation')
parser.add_argument('--num_classes',    type=int, default=2,            help='number of velocity classes')
# Model Saving and reconstruction
parser.add_argument('--output_path', type=str, default='output_hpc/', help='major path for data output')
# Model Parameters
parser.add_argument("--model", type=str, default="vae", help='ae | vae | vae-flow | wae')
parser.add_argument("--beta", type=float, default=1., help='value of beta regularization')
parser.add_argument("--beta_delay", type=int, default=0, help='delay before using beta')
parser.add_argument("--encoder_type", type=str, default="gru", help='mlp | cnn | res-cnn | gru | cnn-gru | hierarchical')
parser.add_argument('--enc_hidden_size', type=int, default=512,         help='do not touch if you do not know')
# PyraPro and vae_mathieu specific parameters: dimensions of the architecture
parser.add_argument('--latent_size', type=int, default=128, help='do not touch if you do not know')
# Optimization parameters
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
# parser.add_argument('--n_runs',             default=5,             type=int,       help='')
# Parse the arguments
args = parser.parse_args()
# Dataset argument
# datasets = ['nottingham', 'maestro', 'bach_chorales', 'fashion_mnist']
# Models grid arguments
model = ['ae', 'vae', 'wae']
# Types of sub-layers in the *AE architectures
encoder_type = ['mlp', 'cnn', 'res-cnn', 'gru', 'cnn-gru']
# Latent sizes
latent_size = [128, 64, 32, 16, 8]
# Beta values
beta_vals = [1.0, 2.0, 8.0]

### TODO = REALLY USE ALL GPUs (DECIDE ON A THING THAT SHOULD BE PARALLELIZED)

# Using list comprehension to compute all possible permutations
res = [[i, j, k, l]     for i in model
                        for j in encoder_type
                        for k in latent_size
                        for l in beta_vals]

full_results = []
full_metadata = []
full_names = []

run_name = 'run_' + str(args.device).replace(':', '_') + '.sh'
for vals in res:
    # Retrieve metadata
    cur_model = vals[0]
    cur_model_id = model.index(vals[0])
    cur_encoder = vals[1]
    cur_encoder_id = encoder_type.index(vals[1])
    cur_latent = vals[2]
    cur_latent_id = latent_size.index(vals[2])
    cur_beta = vals[3]
    cur_beta_id = beta_vals.index(vals[3])
    # Create metadata 
    metadata = [cur_model_id, cur_encoder_id, cur_latent_id, cur_beta_id]
    # Construct path for current model
    model_variants = [args.dataset, args.score_type, args.data_binarize, args.num_classes, args.data_augment, cur_model, cur_encoder, cur_latent, cur_beta, args.enc_hidden_size]
    final_path = args.output_path
    for m in model_variants:
        final_path += str(m) + '_'
    final_path = final_path[:-1] + '/'
    # Check if we have access to the losses
    if (not os.path.exists(final_path + 'losses/_losses.pth')):
        print(final_path)
        print('MISSING')
        continue
    # Load the full losses
    vals = torch.load(final_path + 'losses/_losses.pth');
    full_loss = vals['loss']
    recon_loss = vals['recon_loss']
    # Find the final epoch (early stopping)
    final_epoch = torch.max(torch.nonzero(recon_loss))
    full_loss = full_loss[:final_epoch, :]
    recon_loss = recon_loss[:final_epoch, :]
    if (cur_encoder == 'res-cnn'):
        full_loss /= 3
        recon_loss /= 3
    # Find best epoch (based on valid)
    best_ep = torch.argmin(full_loss[:, 1])
    best_loss = full_loss[best_ep, 2]
    best_recons = recon_loss[best_ep, 2]
    beta_ep = cur_beta * (best_ep / 300.0)
    best_kl = (best_loss - best_recons) * (1. / beta_ep)
    loss_vals = [best_loss, best_kl, best_recons, final_epoch, best_ep]
    # Store current results
    full_results.append(loss_vals)
    full_metadata.append(metadata)
    full_names.append(cur_model + '_' + cur_encoder + '_' + str(cur_latent) + '_' + str(cur_beta))
# Compute the final results
full_results = np.array(full_results)
full_metadata = np.array(full_metadata)
full_names = np.array(full_names)
full_pandas = pandas.DataFrame(np.concatenate([full_results, full_metadata], axis=1), columns=['best_loss', 'best_kl', 'best_recons', 'final_epoch', 'best_ep', 'models', 'encoders', 'latent', 'beta'])

f_write = open('output/figures_hpc/losses_analysis.txt', 'w')
metadata_tabs = [model, encoder_type, latent_size, beta_vals]
titles = ['Model variants', 'Encoder types', 'Latent sizes', 'Beta vals']
names = ['models', 'encoders', 'latent', 'beta']
for c, val in enumerate(metadata_tabs):
    recon_data = []
    loss_data = []
    t = Texttable()
    t.add_row(['Name', 'loss mean', 'loss_std', 'kl_div mean', 'kl_div_std', 'recon_loss mean', 'recon_loss std', 'epoch_mean', 'epoch_std', 'loss_best', 'kl_best', 'recon_best', 'epoch_best'])
    t.set_cols_width([8]*13)
    sns.set_palette("cubehelix", len(val) + 4)
    f_write.write('*'*32 + '\n')
    f_write.write('*' + '\n')
    f_write.write('* Results - ' + names[c] + '\n')
    f_write.write('*' + '\n')
    f_write.write('*'*32 + '\n')
    for c_v, c_str in enumerate(val):
        cur_data = full_results[full_metadata[:, c] == c_v, :]
        names_models = full_names[full_metadata[:, c] == c_v]
        recon_data.append(cur_data[:, 2])
        loss_data.append(cur_data[:, 0])
        cur_data = torch.Tensor(cur_data)
        best_model = torch.argmin(cur_data[:, 0])
        f_write.write('Best - ' + str(c_str) + '\n')
        f_write.write(names_models[best_model] + '\n')
        t.add_row([c_str, torch.mean(cur_data[:, 0]), torch.std(cur_data[:, 0]),
                    torch.mean(cur_data[:, 1]), torch.std(cur_data[:, 1]),
                    torch.mean(cur_data[:, 2]), torch.std(cur_data[:, 2]),
                    torch.mean(cur_data[:, 4]), torch.std(cur_data[:, 4]), 
                    torch.min(cur_data[:, 0]), cur_data[best_model, 1], torch.min(cur_data[:, 2]),
                    cur_data[best_model, 4]])
        for c2, val2 in enumerate(metadata_tabs):
            if (names[c2] == names[c]):
                continue
            t2 = Texttable()
            t2.add_row(['Name', 'loss mean', 'loss_std', 'kl_div mean', 'kl_div_std', 'recon_loss mean', 'recon_loss std', 'epoch_mean', 'epoch_std', 'loss_best', 'kl_best', 'recon_best', 'epoch_best'])
            t2.set_cols_width([8]*13)
            f_write.write('-'*16)
            f_write.write('- Subset ' + str(c_str) + ' - ' + names[c2])
            f_write.write('-'*16 + '\n')
            for c_v2, c_str2 in enumerate(val2):
                cur_data = full_results[np.logical_and (full_metadata[:, c] == c_v, full_metadata[:, c2] == c_v2), :]
                names_models = full_names[np.logical_and(full_metadata[:, c] == c_v, full_metadata[:, c2] == c_v2)]
                cur_data = torch.Tensor(cur_data)
                best_model = torch.argmin(cur_data[:, 0])
                f_write.write('Best - ' + str(c_str) + '+' + str(c_str2) + '\n')
                f_write.write(names_models[best_model] + '\n')
                t2.add_row([str(c_str) + '+' + str(c_str2), torch.mean(cur_data[:, 0]), torch.std(cur_data[:, 0]),
                    torch.mean(cur_data[:, 1]), torch.std(cur_data[:, 1]),
                    torch.mean(cur_data[:, 2]), torch.std(cur_data[:, 2]),
                    torch.mean(cur_data[:, 4]), torch.std(cur_data[:, 4]), 
                    torch.min(cur_data[:, 0]), cur_data[best_model, 1], torch.min(cur_data[:, 2]),
                    cur_data[best_model, 4]])
            f_write.write(t2.draw()) 
            f_write.write('\n')
    f_write.write('*'*32 + '\n')
    f_write.write('* Final table for ' + names[c] + '\n')
    f_write.write('*'*32 + '\n')
    f_write.write(t.draw())
    f_write.write('\n')
f_write.close()
#%%
variant_hash = {'models':model, 'encoders':encoder_type, 'latent':latent_size, 'beta':beta_vals}
plt.figure(figsize=(12, 5))
for plt_name, plt_type in [('loss', 'best_loss'), ('recons', 'best_recons')]:
    for var_type in ['models', 'encoders', 'latent', 'beta']:
        sns_plot = sns.violinplot(data=full_pandas, x=var_type, y=plt_type, scale='width', inner='quartile', linewidth=2.5)
        plt.title(plt_name + ' - ' + var_type)
        sns_plot.set_xticklabels(variant_hash[var_type])
        sns_plot.get_figure().savefig('output/figures_hpc/' + plt_name + '_' + var_type + '_violin.pdf')
        plt.close()
        sns_plot = sns.boxplot(data=full_pandas, x=var_type, y=plt_type, notch=False, linewidth=2.5)
        plt.title(plt_name + ' - ' + var_type)
        sns_plot.set_xticklabels(variant_hash[var_type])
        sns_plot.get_figure().savefig('output/figures_hpc/' + plt_name + '_' + var_type + '_box.pdf')
        plt.close()
        sns_plot = sns.boxenplot(data=full_pandas, x=var_type, y=plt_type, linewidth=2.5)
        plt.title(plt_name + ' - ' + var_type)
        sns_plot.set_xticklabels(variant_hash[var_type])
        sns_plot.get_figure().savefig('output/figures_hpc/' + plt_name + '_' + var_type + '_boxen.pdf')
        plt.close()
        for var_type_2 in ['models', 'encoders', 'latent', 'beta']:
            if (var_type_2 == var_type):
                continue
            ax = sns.boxplot(x=var_type, y=plt_type, hue=var_type_2, data=full_pandas, linewidth=2.5, palette="Set3")
            ax.set_xticklabels(variant_hash[var_type])
            ax.get_figure().savefig('output/figures_hpc/' + plt_name + '_' + var_type + '_' + var_type_2 + '.pdf')
            plt.close()
#ax.legend(encoders)
#ax = sns.swarmplot(x="models", y="best_recons", data=full_pandas, color=".25")

#%%

import random
import pretty_midi
import matplotlib.patches as patches

def interpolation(args, model, dataset, x_a=None, x_b=None, output='output/', fs=25, program=0):
    if (x_a is None):
        x_a, x_b = dataset[random.randint(0, len(dataset) - 1)], dataset[random.randint(0, len(dataset) - 1)]
        x_a, x_b = x_a.to(args.device), x_b.to(args.device)
    # Encode samples to the latent space
    z_a, z_b = model.encode(x_a.unsqueeze(0)), model.encode(x_b.unsqueeze(0))
    # Run through alpha values
    interp = []
    alpha_values = np.linspace(0, 1, args.n_steps)
    for alpha in alpha_values:
        z_interp = (1 - alpha) * z_a[0] + alpha * z_b[0]
        interp.append(model.decode(z_interp))
    # Draw interpolation step by step
    i = 0
    stack_interp = []
    for step in interp:
        if args.num_classes > 1:
            step = torch.argmax(step[0], dim=0)
        stack_interp.append(step)
        # plt.matshow(step.cpu().detach(), alpha=1)
        # plt.title("Interpolation " + str(i))
        # plt.savefig(args.figures_path + "interpolation" + str(i) + ".png")
        # plt.close()
        i += 1
    stack_interp = torch.cat(stack_interp, dim=1)
    # Draw stacked interpolation
    plt.figure()
    plt.matshow(stack_interp.cpu(), alpha=1)
    plt.title("Interpolation")
    plt.savefig(output + "_interpolation.png")
    plt.close()
    # Generate MIDI from interpolation
    pm = pretty_midi.PrettyMIDI()
    notes, frames = stack_interp.shape
    instrument = pretty_midi.Instrument(program=program)
    # Pad 1 column of zeros to acknowledge initial and ending events
    piano_roll = np.pad(stack_interp.cpu().detach(), [(0, 0), (1, 1)], 'constant')
    # Use changes in velocities to find note on/note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)
    # Keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)
    # Do prettier representation
    fig = plt.figure(figsize=(18, 4), dpi=80)
    ax = plt.subplot(1, 1, 1)
    min_pitch = np.inf
    max_pitch = 0
    cmap = plt.get_cmap('inferno', args.n_steps + 3)
    for time, note in zip(*velocity_changes):
        # Use time + 1s because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = 75
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note + args.min_pitch,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
            rect = patches.Rectangle((note_on_time[note] * fs, note + args.min_pitch - 0.5), (time - note_on_time[note]) * fs, 1, linewidth=1.5,
                                 edgecolor='k', facecolor=cmap(int(note_on_time[note] * fs / 64)), alpha=0.8)
            min_pitch = min(min_pitch, note + args.min_pitch)
            max_pitch = max(max_pitch, note + args.min_pitch)
            ax.add_patch(rect) 
    ax.set_ylim([min_pitch - 5, max_pitch + 5])
    ax.set_xticks(np.arange(64, 64*args.n_steps, 64))
    ax.set_xticklabels(np.arange(1, args.n_steps, 1))
    ax.set_xlim([0, time * fs])
    ax.set_xlabel('Interpolated measures')
    ax.set_ylabel('Pitch')
    ax.grid()
    plt.tight_layout()
    plt.savefig(output + "_interpolation.pdf")
    pm.instruments.append(instrument)
    # Write out the MIDI data
    pm.write(output + "_interpolation.mid")

# Here cheat a little
data = torch.load('output/loaders_nottingham_mono_1_2.th')
train_loader, valid_loader, test_loader = data[0], data[1], data[2]
train_set, valid_set, test_set = data[3], data[4], data[5]
train_features, valid_features, test_features = data[6], data[7], data[8]
# Recall minimum pitch
args.min_pitch = train_set.min_p
# Change args
args.device = 'cpu'
args.n_steps = 8
models_compare = ['ae_cnn-gru_128_1.0', 'vae_cnn-gru_16_2.0', 'wae_cnn-gru_64_1.0']
# Select two examples from the dataset
x_a, x_b = test_set[random.randint(0, len(test_set) - 1)], test_set[random.randint(0, len(test_set) - 1)]
for m in models_compare:
    cur_path = 'output_hpc/nottingham_mono_1_2_1_' + m + '_512/models/_full.pth'
    model = torch.load(cur_path, map_location='cpu')
    interpolation(args, model, test_set, x_a, x_b, output='output/figures_hpc/models_' + m)
    
