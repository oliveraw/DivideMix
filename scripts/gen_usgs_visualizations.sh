#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=visualizeDivideMix
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=8000m 
#SBATCH --time=2:00:00
#SBATCH --account=stellayu0
#SBATCH --partition=standard
#SBATCH --output=./logs/%x-%j.log

# The application(s) to execute along with its input arguments and options:
time python Train_usgs.py \
    --train_data_path ../USGS_data/crops/crops_square_32x32 \
    --test_data_path ../USGS_data/crops/crops_square_32x32_val \
    --warmup_epochs 0 \
    --num_epochs 0 \
    --output_dir output-vis \
    --pretrained_net1 output-2-components/net1.pth \
    --pretrained_net2 output-2-components/net2.pth
