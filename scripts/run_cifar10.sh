#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=divideMix
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=8000m 
#SBATCH --time=07:59:55
#SBATCH --account=stellayu0
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --gpu_cmode=shared

# The application(s) to execute along with its input arguments and options:
time python Train_cifar.py --data_path ../cifar-10-batches-py
