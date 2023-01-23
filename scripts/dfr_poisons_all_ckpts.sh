#!/bin/bash
#SBATCH --account=djacobs
#SBATCH --job-name=all-ckpt-dfr
#SBATCH --time=0-8:00:00
#SBATCH --partition=dpart
#SBATCH --qos=default
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --array=7-7
#SBATCH --output=slurm-%j-%x.out
#--SBATCH --dependency=afterok:
#--SBATCH --mail-type=end          
#--SBATCH --mail-type=fail         
#--SBATCH --mail-user=psando@umd.edu

set -x

# DATA_NAMES=('clean' 'ntga' 'error-max' 'untargeted-error-max' 'error-min' 'robust-error-min' 'ar' 'regions-4' 'patches-4' 'patches-8')
declare -a DATA_NAMES=('clean' 'error-min' 'error-max' 'ar' 'ntga' 'robust-error-min' 'regions-4' 'cwrandom' 'random-init-network')
export DATA_NAME=${DATA_NAMES[$SLURM_ARRAY_TASK_ID]}


python3 dfr_evaluate_spurious_all_epochs.py\
    /fs/vulcan-projects/stereo-detection/poison_ckpts/every-epoch-ckpt/linf-poison/${DATA_NAME}
