import os
import argparse
# Argument Parser
parser = argparse.ArgumentParser()
#
parser.add_argument('--midi_path', type=str, default='/fast-1/mathieu/datasets/', help='path to midi folder')
parser.add_argument("--dataset", type=str, default="nottingham", help="maestro | nottingham | bach_chorales | midi_folder")
# Novel arguments to try for winning model
parser.add_argument('--frame_bar',      type=int, default=64,       help='put a power of 2 here')
parser.add_argument('--score_type',     type=str, default='mono',   help='use mono measures or poly ones')
parser.add_argument('--score_sig',      type=str, default='4_4',    help='rhythmic signature to use (use "all" to bypass)')
parser.add_argument('--data_normalize', type=int, default=1,        help='normalize the data')
parser.add_argument('--data_binarize',  type=int, default=1,        help='binarize the data')
parser.add_argument('--data_pitch',     type=int, default=1,        help='constrain pitches in the data')
parser.add_argument('--data_export',    type=int, default=0,        help='recompute the dataset (for debug purposes)')
parser.add_argument('--data_augment',   type=int, default=1,        help='use data augmentation')
# Model Saving and reconstruction
parser.add_argument('--output_path', type=str, default='output/', help='major path for data output')
# Model Parameters
parser.add_argument("--model", type=str, default="vae", help='ae | vae | vae-flow | wae')
parser.add_argument("--beta", type=float, default=1., help='value of beta regularization')
parser.add_argument("--beta_delay", type=int, default=0, help='delay before using beta')
parser.add_argument("--encoder_type", type=str, default="gru", help='mlp | cnn | res-cnn | gru | cnn-gru | hierarchical')
# PyraPro and vae_mathieu specific parameters: dimensions of the architecture
parser.add_argument('--enc_hidden_size', type=int, default=512, help='do not touch if you do not know')
parser.add_argument('--latent_size', type=int, default=128, help='do not touch if you do not know')
parser.add_argument('--cond_hidden_size', type=int, default=1024, help='do not touch if you do not know')
parser.add_argument('--cond_output_dim', type=int, default=512, help='do not touch if you do not know')
parser.add_argument('--dec_hidden_size', type=int, default=512, help='do not touch if you do not know')
parser.add_argument('--num_layers', type=int, default=2, help='do not touch if you do not know')
parser.add_argument('--num_subsequences', type=int, default=8, help='do not touch if you do not know')
parser.add_argument('--num_classes', type=int, default=2, help='number of velocity classes')
parser.add_argument('--initialize', type=int, default=0, help='use initialization on the model')
# Optimization parameters
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--subsample', type=int, default=0, help='train on subset')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--nbworkers', type=int, default=3, help='')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--seed', type=int, default=1, help='random seed')
# Running
parser.add_argument('--n_runs',             default=10,             type=int,       help='')
# Parse the arguments
args = parser.parse_args()
# Dataset argument
datasets = ['nottingham', 'maestro', 'bach_chorales', 'fashion_mnist']
# Models grid arguments
model = ['ae', 'vae', 'vae_mathieu']
# Types of sub-layers in the *AE architectures
encoder_type = ['mlp', 'cnn', 'res_cnn', 'gru', 'cnn_gru', 'hierarchical']
# Latent sizes
latent_size = [256, 128, 64, 32, 16, 4]

# Using list comprehension to compute all possible permutations
res = [[i, j, k] for i in [args.model]
                 for j in [encoder_type]
                 for k in [args.latent_size]]


configurations = {'test': 0, 'full': 1, 'large': 2}
final_config = configurations[args.config_type]

run_name = 'run_' + str(args.device).replace(':', '_') + '.sh'
with open(run_name, 'w') as file:
    for r in range(args.n_runs):
        for vals in res:
            cmd_str = 'python main.py --device ' + args.device
            cmd_str += ' --datadir ' + args.datadir
            cmd_str += ' --output ' + args.output
            cmd_str += ' --dataset ' + args.dataset
            cmd_str += ' --model ' + vals[0]
            cmd_str += ' --encoder_type ' + vals[1]
            cmd_str += ' --latent_size ' + vals[2]
            cmd_str += ' --epochs ' + str(args.epochs)
            print(cmd_str)
            file.write(cmd_str + '\n')
os.system('chmod +x ' + run_name)
