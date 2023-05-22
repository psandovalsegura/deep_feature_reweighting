#!/bin/bash
#SBATCH --account=djacobs
#SBATCH --job-name=dfr-archs
#SBATCH --time=0-8:00:00
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-10
#SBATCH --output=slurm-%j-%x.out
#SBATCH --dependency=afterok:71402
#--SBATCH --mail-type=end          
#--SBATCH --mail-type=fail         
#--SBATCH --mail-user=psando@umd.edu

set -x

declare -a DATA_NAMES=('error-min' 'error-max' 'l2-ar' 'ntga' 'robust-error-min' 'lsp' 'ops-plus-em' 'ops' 'error-min-CW' 'regions-4' 'cwrandom')
export DATA_NAME=${DATA_NAMES[$SLURM_ARRAY_TASK_ID]}

declare -a MODEL_NAMES=('vit-patch-size-4' 'googlenet' 'vgg16')

# perform DFR on all model types
for MODEL_NAME in ${MODEL_NAMES[@]}; do
    python3 dfr_evaluate_spurious_all_epochs.py\
    /fs/vulcan-projects/stereo-detection/unlearnable-ds-neurips-23/dfr-ckpts/${MODEL_NAME}/${DATA_NAME} \
    ${MODEL_NAME} \
    --dataset_name 'cifar10' \
    # --random_init_ckpt
done


