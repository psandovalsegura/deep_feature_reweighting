#!/bin/bash
#SBATCH --account=djacobs
#SBATCH --job-name=all-ckpt-dfr
#SBATCH --time=1-12:00:00
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-3
#SBATCH --output=slurm-%j-%x.out
#--SBATCH --dependency=afterok:
#--SBATCH --mail-type=end          
#--SBATCH --mail-type=fail         
#--SBATCH --mail-user=psando@umd.edu

set -x

# declare -a DATA_NAMES=('error-max' 'error-min'  'l2-ar' 'ntga' 'regions-4' 'cwrandom' 'robust-error-min')
declare -a DATA_NAMES=('lsp' 'ops' 'ops-plus-em' 'error-min-CW')
export DATA_NAME=${DATA_NAMES[$SLURM_ARRAY_TASK_ID]}


python3 dfr_evaluate_spurious_all_epochs.py\
    /fs/vulcan-projects/stereo-detection/unlearnable-ds-neurips-23/dfr-ckpts/${DATA_NAME} \
    # --random_init_ckpt
