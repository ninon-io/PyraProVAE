import os
import argparse
# Argument Parser
parser = argparse.ArgumentParser(description='PyraProVAE')
# Device Information
parser.add_argument('--device', type=str, default='cuda:0', help='device cuda or cpu')
# Data Parameters
parser.add_argument('--midi_path', type=str, default='/fast-1/mathieu/datasets/', help='path to midi folder')
parser.add_argument("--dataset", type=str, default="nottingham", help="maestro | nottingham | bach_chorales | midi_folder")
# Novel arguments
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
parser.add_argument('--latent_size', type=int, default=128, help='do not touch if you do not know')
parser.add_argument('--num_classes', type=int, default=2, help='number of velocity classes')
# Optimization parameters
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--m_runs',             default=8,             type=int,       help='')
parser.add_argument('--n_runs',             default=10,             type=int,       help='')
# Parse the arguments
args = parser.parse_args()

# Ablation on winning model
# Data processing
data_augment = [0, 1]
data_binarize = [0, 1]
data_pitch = [0, 1]
score_sig = ['4_4', 'all']
num_classes = [1, 2]

# Using list comprehension to compute all possible permutations
perm = [[i, j, k, l, m] for i in data_augment
                     for j in data_binarize
                     for k in data_pitch
                     for l in score_sig
                     for m in num_classes]

run_name = 'run_' + str(args.device).replace(':', '_') + '.sh'
with open(run_name, 'w') as file:
    for r in range(args.m_runs):
        for vals in perm:
            cmd_str = 'python main.py --device ' + args.device
            cmd_str += ' --midi_path ' + args.midi_path
            cmd_str += ' --output_path ' + args.output_path
            cmd_str += ' --dataset ' + args.dataset
            cmd_str += ' --model ' + args.model
            cmd_str += ' --encoder_type ' + args.encoder_type
            cmd_str += ' --latent_size ' + args.latent_size
            cmd_str += ' --epochs ' + str(args.epochs)
            cmd_str += ' --data_augment ' + vals[0]
            cmd_str += ' --data_binarize ' + vals[1]
            cmd_str += ' --data_pitch ' + vals[2]
            cmd_str += ' --score_sig ' + vals[3]
            cmd_str += ' --num_classes ' + vals[4]
            print(cmd_str)
            file.write(cmd_str + '\n')
os.system('chmod +x ' + run_name)


# Further trials on winning model, keeping data processing
# Datasets
dataset = ['nottingham', 'maestro', 'bach_chorales']
# Monophony/Polyphony
phony = ['mono', 'poly']
# Learning rates
learning_rate = [0.001, 0.0005, 0.0001]
# Frame bar sizes
frame_bar = [64, 128]


# Using list comprehension to compute all possible permutations
res = [[i, j, k, l] for i in dataset
                    for j in phony
                    for k in learning_rate
                    for l in frame_bar]

run_name = 'run_' + str(args.device).replace(':', '_') + '.sh'
with open(run_name, 'w') as file:
    for r in range(args.n_runs):
        for vals in res:
            cmd_str = 'python main.py --device ' + args.device
            cmd_str += ' --midi_path ' + args.midi_path
            cmd_str += ' --output_path ' + args.output_path
            cmd_str += ' --dataset ' + vals[0]
            cmd_str += ' --model ' + args.model
            cmd_str += ' --encoder_type ' + args.encoder_type
            cmd_str += ' --score_type ' + vals[1]
            cmd_str += ' --lr ' + vals[2]
            cmd_str += ' --frame_bar ' + vals[3]
            cmd_str += ' --latent_size ' + args.latent_size
            cmd_str += ' --epochs ' + str(args.epochs)
            print(cmd_str)
            file.write(cmd_str + '\n')
os.system('chmod +x ' + run_name)
