#!/bin/bash
#SBATCH --account=djacobs
#SBATCH --job-name=dfr-archs-random-init
#SBATCH --time=0-8:00:00
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-0
#SBATCH --output=slurm-%j-%x.out
#SBATCH --dependency=afterok:71364:71376:71388
#--SBATCH --mail-type=end          
#--SBATCH --mail-type=fail         
#--SBATCH --mail-user=psando@umd.edu

set -x

export DATA_NAME='tmp-none'

declare -a MODEL_NAMES=('vgg16' 'vit-patch-size-4' 'googlenet')

# perform DFR on all model types
for MODEL_NAME in ${MODEL_NAMES[@]}; do
    python3 dfr_evaluate_spurious_all_epochs.py\
    /fs/vulcan-projects/stereo-detection/unlearnable-ds-neurips-23/dfr-ckpts/${MODEL_NAME}/${DATA_NAME} \
    ${MODEL_NAME} \
    --dataset_name 'cifar10' \
    --random_init_ckpt
done


