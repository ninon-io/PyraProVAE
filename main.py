from time import time
import argparse
import torch.nn.utils

from learn import Learn
import dataloader
import vae

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
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 1.0)')
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
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

if __name__ == "__main__":
    train_loader, test_loader, train_set, test_set = dataloader.get_data_loader(bar_dir=data_dir, frame_bar=100,
                                                                                batch_size=args.batch_size, export=False)
    learn = Learn(train_loader=train_loader, test_loader=test_loader, train_set=train_set, test_set=test_set,
                  batch_size=args.batch_size, seed=args.seed, lr=args.lr)
    # Load Dataset
    # train_set = dataloader.PianoRollRep(midi_path)
    # test_set = dataloader.PianoRollRep(test_path)
    #
    # # Initialize Dataloader
    # data_loader = torch.utils.data.Dataloader(train_set, batch_size=args.batch_size, num_workers=4, pin_memory=True,
    #                                           shuffle=True, drop_last=True)
    # test_loader = torch.utils.data.Dataloader(test_set, batch_size=args.batch_size, num_workers=4, pin_memory=True,
    #                                           shuffle=True, drop_last=True)

    # Set time
    time0 = time()
    # Initial training of the model
    learn.test()  # First test on randomly initialized data
    for epoch in range(args.epochs):
        print('epoch:' + str(epoch))
        learn.train()
        learn.test()
    # learn.plot()

    print('\nTraining Time in minutes =', (time() - time0) / 60)





