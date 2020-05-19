from time import time
import argparse
import torch.nn.utils
from guppy import hpy

from learn import Learn
import data_loader

h = hpy()

data_dir = 'midi_short_dataset'
# midi_path = "fast-1/ninon/datasets/maestro_folder/train"
# test_path = "fast-1/ninon/datasets/maestro_folder/test"


# get the arguments, if not on command line, the arguments are default
parser = argparse.ArgumentParser(description='Music VAE')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--input_dim', type=int, default=100,
                    help='correspond to the out_channels of the conv (defaults: 123)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')
# parser.add_argument('--local', dest='local', default=False, action='store_true')
# parser.add_argument('--pruning_percent', type=int, default=99.7, metavar='P',
#                     help='percentage of pruning for each cycle (default: 10)')

args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

if __name__ == "__main__":

    data_set, train_loader, test_loader, train_set, test_set = data_loader.get_data_loader(bar_dir=data_dir,
                                                                                           frame_bar=100,
                                                                                           batch_size=args.batch_size,
                                                                                           export=False)

    learn = Learn(train_loader=train_loader, test_loader=test_loader, train_set=train_set, test_set=test_set,
                  batch_size=args.batch_size, seed=args.seed, lr=args.lr)

    # Set time
    time0 = time()
    # Initial training of the model
    # learn.save('/slow-1/ninon/output/models/weights/', 'slow-1/ninon/output/models/entire_model/', epoch=0)
    # learn.save('./models/weights/', './models/entire_model/', epoch=0)
    learn.test(epoch=0)  # First test on randomly initialized data
    for epoch in range(1, args.epochs + 1, 1):
        print('Epoch:' + str(epoch))
        # learn.test_random(data_set, nb_test=1)
        loss_mean, kl_div_mean, recon_loss_mean = learn.train(epoch)
        # learn.test_random(data_set, nb_test=1)
        loss_mean_test, kl_div_mean_test, recon_loss_mean_test = learn.test(epoch)
        # learn.save('/slow-1/ninon/output/models/weights/', 'slow-1/ninon/output/models/entire_model/', epoch)
        learn.save('./models/weights/', './models/entire_model/', epoch)
        # learn.piano_roll_recon('./models/entire_model/' + '_epoch_' + str(epoch) + '.pth') TODO: solve the problem --'

    print('\nTraining Time in minutes =', (time() - time0) / 60)
