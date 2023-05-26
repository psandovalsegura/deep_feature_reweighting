#!/bin/bash
#SBATCH --account=djacobs
#SBATCH --job-name=a2.2oe
#SBATCH --time=1:00:00
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-59
#SBATCH --output /dev/null
#--SBATCH --output=slurm-%j-%x.out
#--SBATCH --dependency=afterok:72935
#--SBATCH --mail-type=end          
#--SBATCH --mail-type=fail         
#--SBATCH --mail-user=psando@umd.edu

set -x

# DATA_NAMES=('cifar100-error-min-SW' 'cifar100-error-max' 'cifar100-error-min-CW' 'cifar100-cwrandom')
export DATA_NAME='cifar100-cwrandom'
export MODEL_NAME='resnet18'

# perform DFR on all model types
python3 dfr_evaluate_spurious_one_epoch.py \
/fs/vulcan-projects/stereo-detection/unlearnable-ds-neurips-23/dfr-ckpts/${DATA_NAME}/${MODEL_NAME}/${DATA_NAME} \
${MODEL_NAME} \
${SLURM_ARRAY_TASK_ID} \
--setup_key 'cifar100' \
--result_path logs-one-epoch/ \


