#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=divideMix
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
#SBATCH --output=./output-dividemix-asym0.3-CE/%x-%j.log

# The application(s) to execute along with its input arguments and options:
time python Train_usgs.py \
    --train_data_path ../USGS_data/crops/crops_square_32x32 \
    --test_data_path ../USGS_data/crops/crops_square_32x32_val \
    --warmup_epochs 0 \
    --num_epochs 0 \
    --n_components_gmm 2 \
    --output_dir output-nodividemix-asym0.3-CE \
    --noise_mode asym \
    --r 0.3\
    --pretrained_net1 output-nodividemix-asym0.3-CE/net1.pth \
    --pretrained_net2 output-nodividemix-asym0.3-CE/net2.pth
