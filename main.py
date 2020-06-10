from time import time
import argparse
import torch.nn.utils
from guppy import hpy
# %%
from learn import Learn
import data_loader
from data_loader import import_dataset
from reconstruction import reconstruction
from vae import VaeModel
from vae import HierarchicalDecoder, HierarchicalEncoder

# For memory checking point if needed
# h = hpy()

# # Argument parser, get the arguments, if not on command line, the arguments are default
parser = argparse.ArgumentParser(description='PyraProVAE')

# Device Information
parser.add_argument('--device', type=str, default='cuda:0', help='device cuda or cpu')

# Data Parameters
parser.add_argument('--midi_path', type=str, default='/fast-1/mathieu/datasets/maestro_folders/train',
                    help='path to midi folder')
parser.add_argument("--valid_size", type=float, default=0.2, help="% of data used in valid set")
parser.add_argument("--dataset", type=str, default="maestro", help="maestro | midi_folder")

# Model Saving
parser.add_argument('--model_path', type=str, default='/slow-2/ninon/pyrapro/models/entire_model/',
                    help='path to the saved model')
parser.add_argument('--weights_path', type=str, default='/slow-2/ninon/pyrapro/models/weights/',
                    help='path to the saved model')

# Model Parameters
parser.add_argument("--model", type=str, default="PyraPro", help='Name of the model')

# PyraPro specific parameters: dimensions of the architecture
parser.add_argument('--input_dim', type=int, default=100, help='do not touch if you do not know')
parser.add_argument('--enc_hidden_size', type=int, default=2048, help='do not touch if you do not know')
parser.add_argument('--latent_size', type=int, default=512, help='do not touch if you do not know')
parser.add_argument('--cond_hidden_size', type=int, default=1024, help='do not touch if you do not know')
parser.add_argument('--cond_output_dim', type=int, default=512, help='do not touch if you do not know')
parser.add_argument('--dec_hidden_size', type=int, default=1024, help='do not touch if you do not know')
parser.add_argument('--num_layers', type=int, default=2, help='do not touch if you do not know')
parser.add_argument('--num_subsequences', type=int, default=8, help='do not touch if you do not know')
parser.add_argument('--seq_lenght', type=int, default=128, help='do not touch if you do not know')

# Optimization parameters
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--test-batch-size', type=int, default=1000, help='input batch size for testing')
parser.add_argument('--frame_bar', type=int, default=100, help='correspond to input dim')
parser.add_argument('--epochs', type=int, default=14, help='number of epochs to train')
parser.add_argument('--nbworkers', type=int, default=3, help='')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--log-interval', type=int, default=10,
                    help='how many batches to wait before logging training status')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')

# Pruning and trimming arguments
# parser.add_argument('--local', dest='local', default=False, action='store_true')
# parser.add_argument('--pruning_percent', type=int, default=99.7, metavar='P',
#                     help='percentage of pruning for each cycle (default: 10)')

# Parse the arguments
args = parser.parse_args()

# Enable CuDNN optimization
if args.device != 'cpu':
    torch.backends.cudnn.benchmark = True

# Handling cuda
args.cuda = not args.device == 'cpu' and torch.cuda.is_available()
args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
# TODO: chose one of the method to handle cuda
use_cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {}

print(10 * '*******')
print('Lovely run info:')
print('The great optimization will be on ' + str(args.device) + '.')
print('The wonderful model is ' + str(args.model) + '.')
print(10 * '*******')

# Data importing
train_loader, valid_loader, test_loader, train_set, test_set, args = import_dataset(args)

# data_set, train_loader, test_loader, train_set, test_set = data_loader.get_data_loader(bar_dir=args.midi_path,
#                                                                                        frame_bar=100,
#                                                                                        batch_size=args.batch_size,
#                                                                                        export=False)

# Model creation
if args.model == 'PyraPro':
    encoder = HierarchicalEncoder(input_dim=args.input_dim, enc_hidden_size=args.enc_hidden_size,
                                  latent_size=args.latent_size, num_layers=args.num_layers)
    decoder = HierarchicalDecoder(input_size=args.input_dim, latent_size=args.latent_size,
                                  cond_hidden_size=args.cond_hidden_size, cond_outdim=args.cond_output_dim,
                                  dec_hidden_size=args.dec_hidden_size, num_layers=args.num_layers,
                                  num_subsequences=args.num_subsequences, seq_length=args.seq_length)
    model = VaeModel(encoder=encoder, decoder=decoder).float()
else:
    print("Oh no, unknown model " + args.model + ".\n")
    exit()
# Send model to the device
model.to(args.device)

# Define learning environment
learn = Learn(args, train_loader=train_loader, test_loader=test_loader, train_set=train_set, test_set=test_set)

# Set time
time0 = time()
# Initial training of the model
learn.save(args, args.weights_path, args.model_path, epoch=0)
learn.test(args, epoch=0)  # First test on randomly initialized data

for epoch in range(1, args.epochs + 1, 1):
    print('Epoch:' + str(epoch))
    loss_mean, kl_div_mean, recon_loss_mean = learn.train(args, epoch)
    loss_mean_test, kl_div_mean_test, recon_loss_mean_test = learn.test(args, epoch)
    learn.save(args, args.weights_path, args.model_path, epoch)
    reconstruction(args, args.midi_path, args.model_path, '/slow-2/ninon/pyrapro/reconstruction/', epoch)

print('\nTraining Time in minutes =', (time() - time0) / 60)
