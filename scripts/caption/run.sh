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
export PYTHONPATH=$PYTHONPATH:./:infer

# python examiners/caption.py --dataset vg --model_path Qwen/Qwen2.5-VL-3B-Instruct --outfile output/vg/caption/Qwen2.5-VL-3B-Instruct.json --num_samples 500
# python examiners/caption.py --dataset vg --model_path Qwen/Qwen2.5-VL-7B-Instruct --outfile output/vg/caption/Qwen2.5-VL-7B-Instruct.json --num_samples 500 &
# python examiners/caption.py --dataset vg --model_path Salesforce/blip2-flan-t5-xl --outfile output/vg/caption/blip2-flan-t5-xl.json --num_samples 500 &
# python examiners/caption.py --dataset vg --model_path Salesforce/blip2-opt-6.7b --outfile output/vg/caption/blip2-opt-6.7b.json --num_samples 500 &
# python examiners/caption.py --dataset vg --model_path Salesforce/blip2-opt-2.7b --outfile output/vg/caption/blip2-opt-2.7b.json --num_samples 500 &
# python examiners/caption.py --dataset vg --model_path llava-hf/llava-1.5-7b-hf --outfile output/vg/caption/llava-1.5-7b-hf.json --num_samples 500 &
# python examiners/caption.py --dataset vg --model_path google/paligemma-3b-mix-224 --outfile output/vg/caption/paligemma-3b-mix-224.json --num_samples 500 &
# python examiners/caption.py --dataset vg --model_path Salesforce/instructblip-vicuna-7b --outfile output/vg/caption/instructblip-vicuna-7b.json --num_samples 500 &
# python examiners/caption.py --dataset vg --model_path THUDM/cogagent-chat-hf --outfile output/vg/caption/cogagent-chat-hf.json --num_samples 500 &
# python examiners/caption.py --dataset vg --model_path Qwen/Qwen2-VL-7B-Instruct --outfile output/vg/caption/Qwen2-VL-7B-Instruct.json --num_samples 500 &
# python examiners/caption.py --dataset vg --model_path Qwen/Qwen2-VL-3B-Instruct --outfile output/vg/caption/Qwen2-VL-3B-Instruct.json --num_samples 500 &
# python examiners/caption.py --dataset vg --model_path AIDC-AI/Ovis2-2B --outfile output/vg/caption/Ovis2-2B.json --num_samples 500 &
# python examiners/caption.py --dataset vg --model_path microsoft/Phi-3.5-vision-instruct --outfile output/vg/caption/Phi-3.5-vision-instruct.json --num_samples 500 &
# python examiners/caption.py --dataset vg --model_path google/gemma-3-4b-it --outfile output/vg/caption/gemma-3-4b-it.json --num_samples 500 &
python examiners/caption.py --dataset vg --model_path OpenGVLab/InternVL3-1B-Instruct --outfile output/vg/caption/InternVL3-1B-Instruct.json --num_samples 500 
python examiners/caption.py --dataset vg --model_path OpenGVLab/InternVL3-2B-Instruct --outfile output/vg/caption/InternVL3-2B-Instruct.json --num_samples 500 
python examiners/caption.py --dataset vg --model_path OpenGVLab/InternVL3-8B-Instruct --outfile output/vg/caption/InternVL3-8B-Instruct.json --num_samples 500
python examiners/caption.py --dataset vg --model_path OpenGVLab/InternVL3-14B-Instruct --outfile output/vg/caption/InternVL3-14B-Instruct.json --num_samples 500

python examiners/caption.py --dataset vg --model_path OpenGVLab/InternVL2_5-8B --outfile output/vg/caption/InternVL2_5-8B.json --num_samples 500 
python examiners/caption.py --dataset vg --model_path OpenGVLab/InternVL2_5-4B --outfile output/vg/caption/InternVL2_5-4B.json --num_samples 500 
python examiners/caption.py --dataset vg --model_path OpenGVLab/InternVL2_5-2B --outfile output/vg/caption/InternVL2_5-2B.json --num_samples 500 
python examiners/caption.py --dataset vg --model_path OpenGVLab/InternVL2_5-1B --outfile output/vg/caption/InternVL2_5-1B.json --num_samples 500 

python examiners/caption.py --dataset vg --model_path OpenGVLab/InternVL2-8B --outfile output/vg/caption/InternVL2-8B.json --num_samples 500 
python examiners/caption.py --dataset vg --model_path OpenGVLab/InternVL2-4B --outfile output/vg/caption/InternVL2-4B.json --num_samples 500 
python examiners/caption.py --dataset vg --model_path OpenGVLab/InternVL2-2B --outfile output/vg/caption/InternVL2-2B.json --num_samples 500 
python examiners/caption.py --dataset vg --model_path OpenGVLab/InternVL2-1B --outfile output/vg/caption/InternVL2-1B.json --num_samples 500 
python examiners/caption.py --dataset vg --model_path data/checkpoints/idefics2-8b-lpoi-list5-10k/final --outfile output/vg/caption/idefics2-8b-lpoi-list5-10k.json --num_samples 500 

python examiners/caption.py --dataset vg --model_path data/checkpoints/LLaVA-RLHF-13b-v1.5-336 --outfile output/vg/caption/LLaVA-RLHF-13b-v1.5-336.json --num_samples 500 


python examiners/caption.py --dataset vg --model_path llava-hf/llava-1.5-7b-hf --outfile output/vg/caption/opera-llava-1.5-7b-hf.json --num_samples 500
CUDA_VISIBLE_DEVICES=7 python examiners/caption.py --dataset vg --model_path opera/llava-1.5 --outfile output/vg/caption/opera-llava-1.5.json --num_samples 500