#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=nodividemixSym
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=8000m 
#SBATCH --time=29:59:55
#SBATCH --account=stellayu0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --gpu_cmode=shared
#SBATCH --output=./output-2components-nodividemix-sym0.2/%x-%j.log

# The application(s) to execute along with its input arguments and options:
time python Train_usgs.py \
    --train_data_path ../USGS_data/crops/crops_square_32x32 \
    --test_data_path ../USGS_data/crops/crops_square_32x32_val \
    --warmup_epochs 200 \
    --num_epochs 200 \
    --n_components_gmm 2 \
    --noise_mode sym \
    --r 0.2 \
    --balanced_softmax \
    --output_dir output-2components-nodividemix-sym0.2
