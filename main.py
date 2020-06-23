from time import time
import argparse
import torch.nn.utils
import numpy as np
# %%
from learn import Learn
from data_loaders.data_loader import import_dataset
from reconstruction import reconstruction
# Import models
from models.vae_pyrapro import VaeModel
from models.vae_pyrapro import HierarchicalEncoder, Decoder
from models.vae_mathieu import VAE_pianoroll, Encoder_pianoroll, Decoder_pianoroll
from texttable import Texttable

# For memory tracking if needed
# h = hpy()

# # Argument parser, get the arguments, if not on command line, the arguments are default
parser = argparse.ArgumentParser(description='PyraProVAE')

# Device Information
parser.add_argument('--device', type=str, default='cuda:0', help='device cuda or cpu')

# Data Parameters
parser.add_argument('--midi_path', type=str, default='/fast-1/mathieu/datasets/maestro_folders/train',
                    help='path to midi folder')
parser.add_argument("--test_size", type=float, default=0.2, help="% of data used in test set")
parser.add_argument("--valid_size", type=float, default=0.2, help="% of data used in valid set")
parser.add_argument("--dataset", type=str, default="maestro", help="maestro | nottingham | bach_chorales | midi_folder")
parser.add_argument("--shuffle_data_set", type=str, default=True, help='')

# Model Saving and reconstruction
parser.add_argument('--model_path', type=str, default='/slow-2/ninon/pyrapro/models_saving/entire_model/',
                    help='path to the saved model')
parser.add_argument('--weights_path', type=str, default='/slow-2/ninon/pyrapro/models_saving/weights/',
                    help='path to the saved model')
parser.add_argument('--figure_reconstruction_path', type=str, default='/slow-2/ninon/pyrapro/reconstruction/',
                    help='path to reconstruction figures')

# Model Parameters
parser.add_argument("--model", type=str, default="PyraPro", help='PyraPro | vae_mathieu | ae')

# PyraPro specific parameters: dimensions of the architecture
parser.add_argument('--input_dim', type=int, default=100, help='do not touch if you do not know')
parser.add_argument('--enc_hidden_size', type=int, default=2048, help='do not touch if you do not know')
parser.add_argument('--latent_size', type=int, default=512, help='do not touch if you do not know')
parser.add_argument('--cond_hidden_size', type=int, default=1024, help='do not touch if you do not know')
parser.add_argument('--cond_output_dim', type=int, default=512, help='do not touch if you do not know')
parser.add_argument('--dec_hidden_size', type=int, default=1024, help='do not touch if you do not know')
parser.add_argument('--num_layers', type=int, default=2, help='do not touch if you do not know')
parser.add_argument('--num_subsequences', type=int, default=8, help='do not touch if you do not know')
parser.add_argument('--seq_length', type=int, default=128, help='do not touch if you do not know')

# Optimization parameters
parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
parser.add_argument('--test-batch-size', type=int, default=1000, help='input batch size for testing')
parser.add_argument('--frame_bar', type=int, default=100, help='correspond to input dim')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
parser.add_argument('--nbworkers', type=int, default=3, help='')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--log-interval', type=int, default=10,
                    help='how many batches to wait before logging training status')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')

# Parse the arguments
args = parser.parse_args()

# Sets the seed for generating random numbers
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Enable CuDNN optimization
if args.device != 'cpu':
    torch.backends.cudnn.benchmark = True

# Handling Cuda
args.cuda = not args.device == 'cpu' and torch.cuda.is_available()
args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
# TODO: chose one of the method to handle cuda
use_cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {}

print(10 * '*******')
print('* Lovely run info:')
print('* Your great optimization will be on ' + str(args.device))
print('* Your wonderful model is ' + str(args.model))
print('* You are using the schwifty ' + str(args.dataset) + ' dataset')
print(10 * '*******')

# Data importing
train_loader, valid_loader, test_loader, train_set, valid_set, test_set, args = import_dataset(args)

# Model creation
if args.model == 'PyraPro':
    encoder = HierarchicalEncoder(input_dim=args.input_dim, enc_hidden_size=args.enc_hidden_size,
                                  latent_size=args.latent_size, num_layers=args.num_layers)
    # decoder = HierarchicalDecoder(input_size=args.input_dim, latent_size=args.latent_size,
    #                              cond_hidden_size=args.cond_hidden_size, cond_outdim=args.cond_output_dim,
    #                              dec_hidden_size=args.dec_hidden_size, num_layers=args.num_layers,
    #                              num_subsequences=args.num_subsequences, seq_length=args.seq_length)
    decoder = Decoder(input_size=args.input_dim, latent_size=args.latent_size,
                      cond_hidden_size=args.cond_hidden_size, cond_outdim=args.cond_output_dim,
                      hidden_size=args.dec_hidden_size, num_layers=args.num_layers,
                      num_subsequences=args.num_subsequences, seq_length=args.seq_length)
    model = VaeModel(encoder=encoder, decoder=decoder).float()
elif args.model == 'vae_mathieu':
    encoder = Encoder_pianoroll(input_dim=args.input_dim, hidden_size=args.enc_hidden_size,
                                  latent_size=args.latent_size, num_layers=args.num_layers)
    decoder = Decoder_pianoroll(input_size=args.input_dim, latent_size=args.latent_size,
                      cond_hidden_size=args.cond_hidden_size, cond_outdim=args.cond_output_dim,
                      dec_hidden_size=args.dec_hidden_size, num_layers=args.num_layers,
                      num_subsequences=args.num_subsequences, seq_length=args.seq_length)
    model = VAE_pianoroll(encoder=encoder, decoder=decoder).float()
else:
    print("Oh no, unknown model " + args.model + ".\n")
    model = None
    exit()
# Send model to the device
model.to(args.device)

# Define learning environment
learn = Learn(args, train_loader=train_loader, validate_loader=valid_loader, test_loader=test_loader,
              train_set=train_set, validate_set=valid_set, test_set=test_set)

# Optimizer and Loss
if args.model == 'PyraPro':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
# if args.model == 'PyraPro':
#     criterion = nn.MSELoss()

# Scheduler
if args.model == 'PyraPro':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                                           verbose=False, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=1e-07, eps=1e-08)
else:
    scheduler = None

# Set time
time0 = time()

# Initial training of the model
learn.save(model, args, epoch=0)

# Initial test
print('INITIAL TEST')
# learn.test(model, args, epoch=0)  # First test on randomly initialized data
print('EPOCH BEGINS')
# Through the epochs
for epoch in range(1, args.epochs + 1, 1):
    print('Epoch:' + str(epoch))  # TODO: print(f"Epoch: {epoch}")
    loss_mean, kl_div_mean, recon_loss_mean = learn.train(model, optimizer, args, epoch)
    loss_mean_validate, kl_div_mean_validate, recon_loss_mean_validate = learn.validate(model, args, epoch)
    scheduler.step(loss_mean_validate)
    loss_mean_test, kl_div_mean_test, recon_loss_mean_test = learn.test(model, args, epoch)
    learn.save(model, args, epoch)
    reconstruction(args, model, epoch)
    # Track on stuffs
    print("*******" * 10)
    print('* Useful & incredible tracking:', 38 * ' ', '*')
    t = Texttable()
    t.add_rows([['Name', 'loss mean', 'kl_div mean', 'recon_loss mean'],
                ['Train', loss_mean, kl_div_mean, recon_loss_mean],
                ['Validate', loss_mean_validate, kl_div_mean_validate, recon_loss_mean_validate],
                ['Test', loss_mean_test, kl_div_mean_test, recon_loss_mean_test]])
    print(t.draw())
    print(10 * '*******')

print('\nTraining Time in minutes =', (time() - time0) / 60)
