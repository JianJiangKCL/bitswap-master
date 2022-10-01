import sys
import os
import subprocess
import argparse
from datetime import datetime
import inspect
import random
import string

# printing lowercase
letters = string.ascii_lowercase
suffix = ''.join(random.choice(letters) for i in range(6))

parser = argparse.ArgumentParser()
parser.add_argument("--script", '-s', type=str, required=True)
parser.add_argument("--mode", type=str, choices=['slurm', 'normal'], default="slurm")
parser.add_argument("--name", '-n', type=str, default=None)
parser.add_argument("--root", type=str, default="commands")
parser.add_argument("--num_gpus", '-gpu', type=int, default=4)
parser.add_argument("--hours", type=int, default=72)
parser.add_argument("--resume", type=int, required=True)
parser.add_argument("--level", type=str, required=True)

args = parser.parse_args()

if args.num_gpus > 1:
    args.hours = 12

else:
    args.hours = 72




# load file
if os.path.exists(args.script):
    with open(args.script) as f:
        command = f.read()

else:
    print(f"{args.script} does not exist.")
    exit()
command_dir = args.root
os.makedirs(command_dir, exist_ok=True)
print(f"temporary commands directory: {command_dir}")

use_ddp = 1 if args.num_gpus > 1 else 0
# run command
total_cpus = 4 * args.num_gpus
if args.mode == "normal":

    os.system(f"bash {args.script}   {args.level} {args.resume}")

elif args.mode == "slurm":

    # build slurm command
    command_prefix = inspect.cleandoc(
        f"""
        #!/bin/bash
        #SBATCH --job-name {args.name}     
        #SBATCH --nodes=1
        #SBATCH --gres gpu:{args.num_gpus}
        #SBATCH --cpus-per-task {total_cpus}
        #SBATCH --time {args.hours}:00:00


        # mail alert at start, end and abortion of execution
        #SBATCH --mail-type=ALL

        # send mail to this address
        #SBATCH --mail-user=jian.jiang@kcl.ac.uk


        # loading conda env
        source ~/.bashrc
        conda activate KM
        # the rest is the scripts

        """
    )

    # write temporary command
    command_path = os.path.join(command_dir, f"command_{suffix}.sh")
    command = command_prefix + command
    with open(command_path, "w") as f:
        f.write(command)

    sbatch_type = None
    if args.num_gpus == 1:
        sbatch_type = f'sbatch -p small --gres=gpu:1 --cpus-per-task={total_cpus}'
    elif args.num_gpus == 4:
        sbatch_type = f'sbatch -p big --gres=gpu:4 --cpus-per-task={total_cpus}'
    elif args.num_gpus == 8:
        sbatch_type = f'sbatch -p big --gres=gpu:8 --cpus-per-task={total_cpus}'

    # run command
    bash_command = f"bash {command_path}  {args.name}"
    print(f"running command: {bash_command}")

    p = subprocess.Popen(f"{sbatch_type} {command_path} {args.level}  {args.resume} ", shell=True,
                         stdout=sys.stdout)  # , stderr=sys.stdout)
    p.wait()
