import os
import argparse
# Argument Parser
parser = argparse.ArgumentParser()
# Device information
parser.add_argument("--datadir",            default="/fast-1/datasets/waveform/",   type=str,       help="Directory to find datasets")
parser.add_argument("--output",             default="/fast-2/philippe/output/",     type=str,       help="Output directory")
parser.add_argument("--dataset",            default="nsynth-10000",                 type=str,       help="mnist | cifar10 | fashion_mnist | cifar100 | toy")
parser.add_argument("--model",              default="",             type=str,       help="mlp | cnn | ae | vae | wae | vae_flow")
parser.add_argument('--epochs',             default=200,            type=int,       help='')
parser.add_argument("--rewind_it",          default=-1,             type=int,       help="Pruning iterations count")
parser.add_argument("--prune_it",           default=2,              type=int,       help="Pruning iterations count")
parser.add_argument("--prune_percent",      default=20,             type=int,       help="Pruning iterations count")
parser.add_argument("--prune",              default="masking",      type=str,       help="masking | trimming | hybrid")
parser.add_argument("--initialize",         default="xavier",       type=str,       help="classic | xavier | kaiming")
parser.add_argument("--device",             default="cpu",          type=str,       help='')
parser.add_argument('--latent_dims',        default=8,              type=int,       help='')
parser.add_argument('--warm_latent',        default=100,            type=int,       help='')
parser.add_argument('--n_runs',             default=10,             type=int,       help='')
parser.add_argument('--config_type',        default='full',         type=str,       help='Preset configuration')
# Parse the arguments
args = parser.parse_args()
# Dataset argument
datasets = ['nottingham', 'maestro', 'bach_chorales', 'fashion_mnist']
# Models grid arguments
model = ['ae', 'vae', 'vae_mathieu']
# Types of sub-layers in the *AE architectures
type_mod = ['mlp', 'cnn', 'res_cnn', 'gru', 'cnn_gru', 'hierarchical']

# Using list comprehension to compute all possible permutations
res = [[i, j, k, l] for i in [args.model]
                    for j in [type_mod]
                    for k in [args.latent_size]
                    for l in prune_selection]


# Set of automatic configurations
class config:
    prune_it        =   [4,     15,     30]
    n_hidden        =   [64,    512,    1024]
    n_layers        =   [3,     4,      6]
    channels        =   [32,    64,     128]
    kernel          =   [5,     5,      5]
    encoder_dims    =   [16,    16,     64]
    eval_interval   =   [1,     45,     50]

configurations = {'test':0, 'full':1, 'large':2}
final_config = configurations[args.config_type]

run_name = 'run_' + str(args.device).replace(':', '_') + '.sh'
with open(run_name, 'w') as file:
    for r in range(args.n_runs):
        for vals in res:
            cmd_str = 'python main.py --device ' + args.device
            cmd_str += ' --datadir ' + args.datadir
            cmd_str += ' --output ' + args.output
            cmd_str += ' --dataset ' + args.dataset
            cmd_str += ' --prune ' + args.prune
            cmd_str += ' --model ' + vals[0]
            cmd_str += ' --prune_reset ' + vals[1]
            cmd_str += ' --prune_scope ' + vals[2]
            cmd_str += ' --prune_selection ' + vals[3]
            cmd_str += ' --prune_percent ' + str(args.prune_percent)
            cmd_str += ' --warm_latent ' + str(args.warm_latent)
            cmd_str += ' --initialize ' + str(args.initialize)
            cmd_str += ' --rewind_it ' + str(args.rewind_it)
            cmd_str += ' --epochs ' + str(args.epochs)
            for n in vars(config):
                if (n[0] != '_'):
                    cmd_str += ' --' + n + ' ' + str(vars(config)[n][final_config])
            cmd_str += ' --k_run ' + str(r)
            print(cmd_str)
            file.write(cmd_str + '\n')
os.system('chmod +x ' + run_name)
