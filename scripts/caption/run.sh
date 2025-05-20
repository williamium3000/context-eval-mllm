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
# python examiners/caption.py --dataset vg --outfile output/vg/caption/caption.json --num_samples 100
python examiners/caption.py --dataset vg --model_path Qwen/Qwen2.5-VL-3B-Instruct --outfile output/vg/caption/Qwen2.5-VL-3B-Instruct.json --num_samples 100 &
python examiners/caption.py --dataset vg --model_path Qwen/Qwen2.5-VL-7B-Instruct --outfile output/vg/caption/Qwen2.5-VL-7B-Instruct.json --num_samples 100 &
python examiners/caption.py --dataset vg --model_path Salesforce/blip2-flan-t5-xl --outfile output/vg/caption/blip2-flan-t5-xl.json --num_samples 100 &
python examiners/caption.py --dataset vg --model_path llava-hf/llava-1.5-7b-hf --outfile output/vg/caption/llava-1.5-7b-hf.json --num_samples 100 &