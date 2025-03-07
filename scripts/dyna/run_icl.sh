#!/bin/bash -l
#SBATCH --job-name=test
#SBATCH --time=4:0:0
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output "slurm_logs/slurm-%j.out"

# mkdir -p slurm_logs
# conda activate llava

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0

# python examiners/dyna_conv_icl_certainty.py --model_path llava-hf/llava-1.5-7b-hf --dataset vg --p_mode None --outfile output/vg/dyna/icl_certainty.json
python examiners/dyna_conv_icl.py --model_path llava-hf/llava-1.5-7b-hf --dataset vg --p_mode None --outfile output/vg/dyna/icl.json
