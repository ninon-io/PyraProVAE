import os
import argparse
# Argument Parser
parser = argparse.ArgumentParser()
# Device Information
parser.add_argument('--device',         type=str,   default='cuda',     help='device cuda or cpu')
# Data information
parser.add_argument('--midi_path',      type=str,   default='/scratch/esling/datasets/', help='path to midi folder')
parser.add_argument("--dataset",        type=str,   default="nottingham", help="maestro | nottingham | bach_chorales | midi_folder")
# Model Saving and reconstruction
parser.add_argument('--output_path',    type=str,   default='/scratch/esling/output/', help='major path for data output')
# Model Parameters
parser.add_argument("--model",          type=str,   default="vae",      help='ae | vae | vae-flow | wae')
parser.add_argument("--beta",           type=float, default=1.,         help='value of beta regularization')
parser.add_argument("--beta_delay",     type=int,   default=0,          help='delay before using beta')
parser.add_argument("--encoder_type",   type=str,   default="gru",      help='mlp | cnn | res-cnn | gru | cnn-gru | hierarchical')
# PyraPro and vae_mathieu specific parameters: dimensions of the architecture
parser.add_argument('--latent_size',    type=int,   default=128,        help='do not touch if you do not know')
# Optimization parameters
parser.add_argument('--epochs',         type=int,   default=300,        help='number of epochs to train')
parser.add_argument('--machine',        type=str,   default='cedar',    help='Machine on which we are computing')
parser.add_argument('--time',           type=str,   default='0-11:59',  help='Machine on which we are computing')
# Parse the arguments
args = parser.parse_args()
# Dataset argument
# datasets = ['nottingham', 'maestro', 'bach_chorales', 'fashion_mnist']
# Models grid arguments
model = ['ae', 'vae', 'wae']
# Types of sub-layers in the *AE architectures
encoder_type = ['mlp', 'cnn', 'res_cnn', 'gru', 'cnn_gru', 'hierarchical']
# Latent sizes
latent_size = [256, 128, 64, 32, 16, 4]
# Beta values
beta_vals = [1.0, 2.0, 5.0, 10.0]


# Scripts folder
if not os.path.exists('scripts/output/'):
    os.makedirs('./scripts/output/')
    
def write_basic_script(file, args, out_f="%N-%j.out"):
    file.write("#!/bin/bash\n")
    file.write("#SBATCH --gres=gpu:1            # Request GPU generic resources\n")
    if (args.machine == 'cedar'):
        file.write("#SBATCH --cpus-per-task=6   # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.\n")
        file.write("#SBATCH --mem=32000M        # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.\n")
    else:
        file.write("#SBATCH --cpus-per-task=16   # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.\n")
        file.write("#SBATCH --mem=64000M         # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.\n")
    file.write("#SBATCH --time=%s\n"%(args.time))
    file.write("#SBATCH --output=" + out_f + "\n")
    file.write("\n")
    file.write("module load python/3.7\n")
    if (args.machine != 'local'):
        file.write("virtualenv --no-download $SLURM_TMPDIR/env\n")
        file.write("source $SLURM_TMPDIR/env/bin/activate\n")
        file.write("pip install --no-index --upgrade pip\n")
        file.write("pip install --no-index -r requirements.txt\n")
        # This is ugly as fuck ... but mandatory
        file.write("pip install ~/scratch/python_libs/pretty_midi-0.2.9.tar.gz\n")
        #file.write("pip install ~/scratch/python_libs/SoundFile-0.10.3.post1-py2.py3-none-any.whl\n")
        #file.write("pip install ~/scratch/python_libs/resampy-0.2.2.tar.gz\n")
        #file.write("pip install ~/scratch/python_libs/librosa-0.7.2.tar.gz\n")
        #file.write("pip install ~/scratch/python_libs/lmdb-0.98.tar.gz\n")
        #file.write("pip install ~/scratch/python_libs/mir_eval-0.6.tar.gz\n")
        file.write("cd $SLURM_TMPDIR\n")
    else:
        file.write("source $HOME/env/bin/activate\n")
    file.write("\n")
    file.write("cd /scratch/esling/ninon/\n")

# Using list comprehension to compute all possible permutations
res = [[i, j, k, l]     for i in model
                        for j in encoder_type
                        for k in latent_size
                        for l in beta_vals]

run_name = 'run_' + str(args.dataset) + '.sh'
cpt = 1
with open(run_name, 'w') as file:
    #for r in range(args.n_runs):
    for vals in res:
        model_vals = vals[0] + '_' + vals[1] + '_' + str(vals[2]) + '_' + str(vals[3])
        # Write the original script file
        final_script = 'scripts/sc_'  + model_vals + '.sh'
        f_script = open(final_script, 'w')
        write_basic_script(f_script, args, 'output/out_'  + model_vals)
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
        f_script.write(cmd_str + '\n')
        f_script.close()
        file.write('sbatch ' + final_script + '\n')
        cpt += 1
os.system('chmod +x ' + run_name)
