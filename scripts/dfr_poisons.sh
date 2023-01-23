#!/bin/bash
#SBATCH --account=djacobs
#SBATCH --job-name=dfr
#SBATCH --time=0-8:00:00
#SBATCH --partition=dpart
#SBATCH --qos=default
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --array=7-8
#SBATCH --output=slurm-%j-%x.out
#--SBATCH --dependency=afterok:
#--SBATCH --mail-type=end          
#--SBATCH --mail-type=fail         
#--SBATCH --mail-user=psando@umd.edu

set -x

# DATA_NAMES=('clean' 'ntga' 'error-max' 'untargeted-error-max' 'error-min' 'robust-error-min' 'ar' 'regions-4' 'patches-4' 'patches-8')
declare -a DATA_NAMES=('clean' 'error-min' 'error-max' 'ar' 'ntga' 'robust-error-min' 'regions-4' 'cwrandom' 'random-init-network')
export DATA_NAME=${DATA_NAMES[$SLURM_ARRAY_TASK_ID]}

declare -a PERCENTS=(0.05 0.1 0.3 0.5 1.0)

for PERCENT in ${PERCENTS[@]}; do
        python3 dfr_evaluate_spurious.py\
                ${DATA_NAME}\
                --percent_train=${PERCENT}\
                --ckpt_path=/fs/vulcan-projects/stereo-detection/poison_ckpts/linf-poison/${DATA_NAME}
done
