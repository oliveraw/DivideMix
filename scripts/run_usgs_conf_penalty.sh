#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=divideMix
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=8000m 
#SBATCH --time=39:59:55
#SBATCH --account=stellayu0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --gpu_cmode=shared
#SBATCH --output=./logs/%x-%j.log

# The application(s) to execute along with its input arguments and options:
time python Train_usgs.py \
    --train_data_path ../USGS_data/crops/crops_square_32x32 \
    --test_data_path ../USGS_data/crops/crops_square_32x32_val \
    --warmup_epochs 5 \
    --n_components_gmm 2 \
    --output_dir output-2-components-conf-penalty

time python Train_usgs.py \
    --train_data_path ../USGS_data/crops/crops_square_32x32 \
    --test_data_path ../USGS_data/crops/crops_square_32x32_val \
    --warmup_epochs 5 \
    --n_components_gmm 3 \
    --output_dir output-3-components-conf-penalty

time python Train_usgs.py \
    --train_data_path ../USGS_data/crops/crops_square_32x32 \
    --test_data_path ../USGS_data/crops/crops_square_32x32_val \
    --warmup_epochs 5 \
    --n_components_gmm 4 \
    --output_dir output-4-components-conf-penalty
