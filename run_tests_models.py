import os
import argparse
# Argument Parser
parser = argparse.ArgumentParser()
# Device Information
parser.add_argument('--device', type=str, default='cuda:0', help='device cuda or cpu')
# Data information
parser.add_argument('--midi_path', type=str, default='/fast-1/mathieu/datasets/', help='path to midi folder')
parser.add_argument("--dataset", type=str, default="nottingham", help="maestro | nottingham | bach_chorales | midi_folder")
# Model Saving and reconstruction
parser.add_argument('--output_path', type=str, default='output/', help='major path for data output')
# Model Parameters
parser.add_argument("--model", type=str, default="vae", help='ae | vae | vae-flow | wae')
parser.add_argument("--beta", type=float, default=1., help='value of beta regularization')
parser.add_argument("--beta_delay", type=int, default=0, help='delay before using beta')
parser.add_argument("--encoder_type", type=str, default="gru", help='mlp | cnn | res-cnn | gru | cnn-gru | hierarchical')
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
encoder_type = ['mlp', 'cnn', 'res-cnn', 'gru', 'cnn-gru', 'hierarchical']
# Latent sizes
latent_size = [256, 128, 64, 32, 16, 4]
# Beta values
beta_vals = [1.0, 2.0, 5.0, 10.0]

### TODO = REALLY USE ALL GPUs (DECIDE ON A THING THAT SHOULD BE PARALLELIZED)

# Using list comprehension to compute all possible permutations
res = [[i, j, k, l]     for i in model
                        for j in encoder_type
                        for k in latent_size
                        for l in beta_vals]

run_name = 'run_' + str(args.device).replace(':', '_') + '.sh'
with open(run_name, 'w') as file:
    #for r in range(args.n_runs):
    for vals in res:
        cmd_str = 'python main.py --device ' + args.device
        cmd_str += ' --midi_path ' + args.midi_path
        cmd_str += ' --output_path ' + args.output_path
        cmd_str += ' --dataset ' + args.dataset
        cmd_str += ' --model ' + vals[0]
        cmd_str += ' --encoder_type ' + vals[1]
        cmd_str += ' --latent_size ' + str(vals[2])
        cmd_str += ' --beta ' + str(vals[3])
        cmd_str += ' --epochs ' + str(args.epochs)
        #cmd_str += ' --k_run ' + str(r)
        print(cmd_str)
        file.write(cmd_str + '\n')
os.system('chmod +x ' + run_name)
