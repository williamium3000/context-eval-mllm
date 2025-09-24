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
export CUDA_VISIBLE_DEVICES=1

conda activate  /mnt/data/ztw_project/mmeval_envs/qwenvl2d5
# python examiner/dyna_conv_v2.py --dataset vg --model_path Qwen/Qwen2.5-VL-3B-Instruct --outfile output/vg/dyna-v2/Qwen2.5-VL-3B-Instruct.json --num_samples 100
# python examiner/dyna_conv_v2.py --dataset vg --model_path Qwen/Qwen2.5-VL-7B-Instruct --outfile output/vg/dyna-v2/Qwen2.5-VL-7B-Instruct.json --num_samples 100

# python examiner/dyna_conv_v2.py --dataset vg --model_path llava-hf/llava-1.5-7b-hf --outfile output/vg/dyna-v2/llava-1.5-7b-hf.json --num_samples 100
conda activate /home/ubuntu/share/miniconda3/envs/opera
# python examiner/dyna_conv_v2.py --dataset vg --model_path opera/llava-1.5  --outfile output/vg/dyna-v2/opera-llava-1.5.json --num_samples 100

# conda activate /mnt/data/cogbench_envs/phi4
python examiner/dyna_conv_v2.py --dataset vg --model_path OpenGVLab/InternVL2_5-8B  --outfile output/vg/dyna-v2/InternVL2_5-8B.json --num_samples 100
python examiner/dyna_conv_v2.py --dataset vg --model_path OpenGVLab/InternVL3-8B-Instruct  --outfile output/vg/dyna-v2/InternVL3-8B-Instruct.json --num_samples 100
python examiner/dyna_conv_v2.py --dataset vg --model_path OpenGVLab/InternVL2-8B  --outfile output/vg/dyna-v2/InternVL2-8B.json --num_samples 100


python examiner/dyna_conv_v2.py --dataset vg --model_path data/checkpoints/LLaVA-RLHF-13b-v1.5-336 --outfile output/vg/dyna-v2/LLaVA-RLHF-13b-v1.5-336.json --num_samples 100
# python examiner/dyna_conv_v2.py --dataset vg --model_path microsoft/Phi-3.5-vision-instruct  --outfile output/vg/dyna-v2/Phi-3.5-vision-instruct.json --num_samples 100
# python examiner/dyna_conv_v2.py --dataset vg --model_path Salesforce/blip2-flan-t5-xl  --outfile output/vg/dyna-v2/blip2-flan-t5-xl.json --num_samples 100
# python examiner/dyna_conv_v2.py --dataset vg --model_path Salesforce/blip2-opt-2.7b  --outfile output/vg/dyna-v2/blip2-opt-2.7b.json --num_samples 100

# python examiner/context.py --dataset vg --icls icls/vg_context_icl.json --model_path Qwen/Qwen2.5-VL-3B-Instruct --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL_CERTAINTY_CONTEXT --outfile output/vg/context/CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL_CERTAINTY_CONTEXT.json --num_samples 5
# python examiner/context.py --dataset vg --model_path Qwen/Qwen2.5-VL-3B-Instruct --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL_CERTAINTY_CONTEXT_UNANSWERABLE_ADVERSARIA_REVISED --outfile output/vg/context/CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL_CERTAINTY_CONTEXT_UNANSWERABLE_ADVERSARIA_REVISED.json --num_samples 5
# python examiner/context.py --dataset vg --icls icls/vg_context_icl.json --model_path Qwen/Qwen2.5-VL-3B-Instruct --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_VG_SIMGPLE_INTERROGATE_CERTAINTY --outfile output/vg/dyna/context_icls.json --num_samples 5
# python examiners/dyna_conv.py --model_path llava-hf/llava-1.5-7b-hf --dataset vg --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_VG_UNANSWERABLE --outfile output/vg/dyna/CONV_MODEL_PERSPECTIVE_PROMPT_VG_UNANSWERABLE.json
# python examiners/dyna_conv.py --dataset vg --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_VG_SIMGPLE --outfile output/vg/dyna/CONV_MODEL_PERSPECTIVE_PROMPT_VG_SIMGPLE.json
# python examiners/dyna_conv_json.py --dataset vg --p_mode CONV_COVERAGE_PROMPT_CERTAINTY_WITH_ANSWER_VG --outfile output/vg/dyna/CONV_COVERAGE_PROMPT_CERTAINTY_WITH_ANSWER_VG.json

# python examiners/dyna_conv.py --model_path llava-hf/llava-1.5-7b-hf --dataset coco --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_COCO_SIMGPLE_INTERROGATE --outfile output/vg/dyna/coco/llava-1.5-7b-hf/CONV_MODEL_PERSPECTIVE_PROMPT_COCO_SIMGPLE_INTERROGATE.json --num_samples 5
# python examiners/dyna_conv.py --model_path llava-hf/llava-1.5-7b-hf --dataset coco --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_COCO_UNANSWERABLE --outfile output/vg/dyna/coco/llava-1.5-7b-hf/CONV_MODEL_PERSPECTIVE_PROMPT_COCO_UNANSWERABLE.json --num_samples 5