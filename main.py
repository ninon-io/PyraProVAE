    # %%
import os
import argparse
import torch.nn as nn
import torch.nn.utils
import numpy as np
from time import time
from texttable import Texttable
# Personnal imports
from learn import Learn
from data_loaders.data_loader import import_dataset
from reconstruction import reconstruction, sampling, interpolation
# Import encoders
from models.encoders import EncoderMLP, DecoderMLP, EncoderCNN, DecoderCNN
from models.encoders import EncoderGRU, DecoderGRU, EncoderCNNGRU, DecoderCNNGRU, DecoderCNNGRUEmbedded
from models.encoders import EncoderHierarchical, DecoderHierarchical
# Import model variants
from models.ae import AE, VAE, WAE
# Import initializer
from utils import init_classic

# %%
# -----------------------------------------------------------
#
# Argument parser, get the arguments, if not on command line, the arguments are default
#
# -----------------------------------------------------------
parser = argparse.ArgumentParser(description='PyraProVAE')
# Device Information
parser.add_argument('--device',         type=str, default='cuda:2',     help='device cuda or cpu')
# Data Parameters
parser.add_argument('--midi_path',      type=str, default='/Users/esling/datasets/symbolic/', help='path to midi folder')
parser.add_argument("--test_size",      type=float, default=0.2,        help="% of data used in test set")
parser.add_argument("--valid_size",     type=float, default=0.2,        help="% of data used in valid set")
parser.add_argument("--dataset",        type=str, default="nottingham", help="maestro | nottingham | bach_chorales | combo")
parser.add_argument("--shuffle_data_set", type=int, default=1,          help='')
# Novel arguments
parser.add_argument('--frame_bar',      type=int, default=64,           help='put a power of 2 here')
parser.add_argument('--score_type',     type=str, default='mono',       help='use mono measures or poly ones')
parser.add_argument('--score_sig',      type=str, default='4_4',        help='rhythmic signature to use (use "all" to bypass)')
parser.add_argument('--data_normalize', type=int, default=1,            help='normalize the data')
parser.add_argument('--data_binarize',  type=int, default=1,            help='binarize the data')
parser.add_argument('--data_pitch',     type=int, default=1,            help='constrain pitches in the data')
parser.add_argument('--data_export',    type=int, default=0,            help='recompute the dataset (for debug purposes)')
parser.add_argument('--data_augment',   type=int, default=1,            help='use data augmentation')
# Model Saving and reconstruction
parser.add_argument('--output_path',    type=str, default='output/', help='major path for data output')
# Model Parameters
parser.add_argument("--model",          type=str, default="vae",        help='ae | vae | vae-flow | wae')
parser.add_argument("--encoder_type",   type=str, default="cnn-gru",    help='mlp | cnn | res-cnn | gru | cnn-gru | hierarchical')
parser.add_argument("--beta",           type=float, default=2.,         help='value of beta regularization')
parser.add_argument("--beta_delay",     type=int, default=0,            help='delay before using beta')
# PyraPro and vae_mathieu specific parameters: dimensions of the architecture
parser.add_argument('--enc_hidden_size', type=int, default=512,         help='do not touch if you do not know')
parser.add_argument('--latent_size',    type=int, default=64,          help='do not touch if you do not know')
parser.add_argument('--cond_hidden_size', type=int, default=1024,       help='do not touch if you do not know')
parser.add_argument('--cond_output_dim', type=int, default=512,         help='do not touch if you do not know')
parser.add_argument('--dec_hidden_size', type=int, default=512,         help='do not touch if you do not know')
parser.add_argument('--num_layers',     type=int, default=2,            help='do not touch if you do not know')
parser.add_argument('--num_subsequences', type=int, default=8,          help='do not touch if you do not know')
parser.add_argument('--num_classes',    type=int, default=2,            help='number of velocity classes')
parser.add_argument('--initialize',     type=int, default=0,            help='use initialization on the model')
# Optimization parameters
parser.add_argument('--batch_size',     type=int, default=64,           help='input batch size')
parser.add_argument('--subsample',      type=int, default=0,            help='train on subset')
parser.add_argument('--epochs',         type=int, default=300,          help='number of epochs to train')
parser.add_argument('--early_stop',     type=int, default=42,           help='')
parser.add_argument('--nbworkers',      type=int, default=3,            help='')
parser.add_argument('--lr',             type=float, default=0.0001,     help='learning rate')
parser.add_argument('--seed',           type=int, default=1,            help='random seed')
# Reconstruction parameters
parser.add_argument('--n_steps',        type=int, default=11,           help='number of steps for interpolation')
parser.add_argument('--nb_samples',     type=int, default=8,            help='number of samples to decode from latent space')
# Parse the arguments
args = parser.parse_args()

# %%
# -----------------------------------------------------------
#
# Base setup section
#
# -----------------------------------------------------------
# Sets the seed for generating random numbers
torch.manual_seed(args.seed)
np.random.seed(args.seed)
# Enable CuDNN optimization
if args.device != 'cpu':
    torch.backends.cudnn.benchmark = True
# Handling Cuda
args.cuda = not args.device == 'cpu' and torch.cuda.is_available()
args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
# Print info
print(10 * '*******')
print('* Lovely run info:')
print('* Your great optimization will be on ' + str(args.device))
print('* Your wonderful model is ' + str(args.model))
print('* You are using the schwifty ' + str(args.dataset) + ' dataset')
print(10 * '*******')
# Handling directories
model_variants = [args.dataset, args.score_type, args.data_binarize, args.num_classes, args.data_augment, args.model, args.encoder_type, args.latent_size, args.beta, args.enc_hidden_size]
args.final_path = args.output_path
for m in model_variants:
    args.final_path += str(m) + '_'
args.final_path = args.final_path[:-1] + '/'
if os.path.exists(args.final_path):
    os.system('rm -rf ' + args.final_path + '/*')
else:
    os.makedirs(args.final_path)
# Create all sub-folders
args.model_path = args.final_path + 'models/'
args.losses_path = args.final_path + 'losses/'
args.tensorboard_path = args.final_path + 'tensorboard/'
args.weights_path = args.final_path + 'weights/'
args.figures_path = args.final_path + 'figures/'
args.midi_results_path = args.final_path + 'midi/'
for p in [args.model_path, args.losses_path, args.tensorboard_path, args.weights_path, args.figures_path, args.midi_results_path]:
    os.makedirs(p)
# Ensure coherence of classes parameters
if args.data_binarize and args.num_classes > 1:
    args.num_classes = 2

# %%
# -----------------------------------------------------------
#
# Base setup section
#
# -----------------------------------------------------------
# Data importing
print('[Importing dataset]')
train_loader, valid_loader, test_loader, train_set, valid_set, test_set, args = import_dataset(args)
args.min_pitch = test_set.min_p

# %%
# -----------------------------------------------------------
#
# Model and layers creation
#
# -----------------------------------------------------------
# Model creation
print('[Creating encoder and decoder]')
# Here select between different encoders and decoders
if args.encoder_type == 'mlp':
    encoder = EncoderMLP(args)
    decoder = DecoderMLP(args)
elif args.encoder_type == 'cnn':
    args.type_mod = 'normal'
    encoder = EncoderCNN(args)
    args.cnn_size = encoder.cnn_size
    decoder = DecoderCNN(args)
elif args.encoder_type == 'res-cnn':
    args.type_mod = 'residual'
    encoder = EncoderCNN(args)
    args.cnn_size = encoder.cnn_size
    decoder = DecoderCNN(args)
elif args.encoder_type == 'gru':
    encoder = EncoderGRU(args)
    decoder = DecoderGRU(args)
elif args.encoder_type == 'cnn-gru':
    args.type_mod = 'normal'
    encoder = EncoderCNNGRU(args)
    args.cnn_size = encoder.cnn_size
    decoder = DecoderCNNGRU(args)
elif args.encoder_type == 'cnn-gru-embed':
    args.type_mod = 'normal'
    encoder = EncoderCNNGRU(args)
    args.cnn_size = encoder.cnn_size
    decoder = DecoderCNNGRUEmbedded(args)
elif args.encoder_type == 'hierarchical':
    encoder = EncoderHierarchical(args)
    decoder = DecoderHierarchical(args)
print('[Creating model]')
# Then select different models
if args.model == 'ae':
    model = AE(encoder, decoder, args).float()
elif args.model == 'vae':
    model = VAE(encoder, decoder, args).float()
elif args.model == 'wae':
    model = WAE(encoder, decoder, args).float()
else:
    print("Oh no, unknown model " + args.model + ".\n")
    exit()
# Send model to the device
model.to(args.device)
# Initialize the model weights
print('[Initializing weights]')
if args.initialize:
    model.apply(init_classic)

# %%
# -----------------------------------------------------------
#
# Optimizer
#
# -----------------------------------------------------------
print('[Creating optimizer]')
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
# Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                                       verbose=False, threshold=0.0001, threshold_mode='rel',
                                                       cooldown=0, min_lr=1e-07, eps=1e-08)
# Learning class
learn = Learn(args, train_loader=train_loader, validate_loader=valid_loader, test_loader=test_loader,
              train_set=train_set, validate_set=valid_set, test_set=test_set)

# %%
# -----------------------------------------------------------
#
# Losses functions
#
# -----------------------------------------------------------
print('[Creating criterion]')
# Losses
if args.model in ['ae', 'vae', 'wae', 'vae-flow']:
    criterion = nn.MSELoss()
if args.num_classes > 1:
    criterion = nn.NLLLoss(reduction='sum')

# %%
# -----------------------------------------------------------
#
# Training loop
#
# -----------------------------------------------------------
# Set time
time0 = time()
# Initial test
print('[Initial evaluation]')
# learn.test(model, args, epoch=0)  # First test on randomly initialized data
print('[Starting main training]')
# Set losses
losses = torch.zeros(args.epochs + 1, 3)
recon_losses = torch.zeros(args.epochs + 1, 3)
# Set minimum to infinity
cur_best_valid = np.inf
cur_best_valid_recons = np.inf
# Set early stop
early_stop = 0
# Through the epochs
for epoch in range(1, args.epochs + 1, 1):
    print(f"Epoch: {epoch}")
    # Training epoch
    loss_mean, kl_div_mean, recon_loss_mean = learn.train(model, optimizer, criterion, args, epoch)
    # Validate epoch
    loss_mean_validate, kl_div_mean_validate, recon_loss_mean_validate = learn.validate(model, criterion,  args, epoch)
    # Step for learning rate
    scheduler.step(loss_mean_validate)
    # Test model
    loss_mean_test, kl_div_mean_test, recon_loss_mean_test = learn.test(model, criterion, args, epoch)
    # Compare input data and reconstruction
    if (epoch % 25 == 0):
        reconstruction(args, model, epoch, test_set)
    # Gather losses
    loss_list = [loss_mean, loss_mean_validate, loss_mean_test]
    for counter, loss in enumerate(loss_list):
        losses[epoch - 1, counter] = loss
    # Gather reconstruction losses
    recon_loss_list = [recon_loss_mean, recon_loss_mean_validate, recon_loss_mean_test]
    for counter, loss in enumerate(recon_loss_list):
        recon_losses[epoch - 1, counter] = loss
    # Save losses
    torch.save({
        'loss': losses,
        'recon_loss': recon_losses,
    }, args.losses_path + '_losses.pth')
    # Save best weights (mean validation loss)
    if recon_loss_mean_validate < cur_best_valid_recons:
        cur_best_valid_recons = recon_loss_mean_validate
        learn.save(model, args, 'reconstruction')
    # Save best weights (mean validation loss)
    if loss_mean_validate < cur_best_valid:
        cur_best_valid = loss_mean_validate
        learn.save(model, args, 'full')
        early_stop = 0
    elif args.early_stop > 0:
        early_stop += 1
        if early_stop > args.early_stop:
            print('[Model stopped early]')
            break
    # Track on stuffs
    print("*******" * 10)
    print('* Useful & incredible tracking:')
    t = Texttable()
    t.add_rows([['Name', 'loss mean', 'kl_div mean', 'recon_loss mean'],
                ['Train', loss_mean, kl_div_mean, recon_loss_mean],
                ['Validate', loss_mean_validate, kl_div_mean_validate, recon_loss_mean_validate],
                ['Test', loss_mean_test, kl_div_mean_test, recon_loss_mean_test]])
    print(t.draw())
    print(10 * '*******')
print('\nTraining Time in minutes =', (time() - time0) / 60)

#%% -----------------------------------------------------------
#
# Evaluate stuffs
#
# -----------------------------------------------------------
print('[Evaluation]')
# Reload best performing model
model = torch.load(args.model_path + '_' + 'full' + '.pth', map_location=args.device)
# Sample random point from latent space
sampling(args, model)
# Interpolation between two inputs
interpolation(args, model, test_set)

