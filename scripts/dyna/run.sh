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


# python examiners/dyna_conv.py --dataset vg --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_VG_SIMGPLE_INTERROGATE_CERTAINTY --outfile output/vg/dyna/CONV_MODEL_PERSPECTIVE_PROMPT_VG_SIMGPLE_INTERROGATE_CERTAINTY.json
# python examiners/dyna_conv.py --model_path llava-hf/llava-1.5-7b-hf --dataset vg --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_VG_UNANSWERABLE --outfile output/vg/dyna/CONV_MODEL_PERSPECTIVE_PROMPT_VG_UNANSWERABLE.json
# python examiners/dyna_conv.py --dataset vg --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_VG_SIMGPLE --outfile output/vg/dyna/CONV_MODEL_PERSPECTIVE_PROMPT_VG_SIMGPLE.json
# python examiners/dyna_conv_json.py --dataset vg --p_mode CONV_COVERAGE_PROMPT_CERTAINTY_WITH_ANSWER_VG --outfile output/vg/dyna/CONV_COVERAGE_PROMPT_CERTAINTY_WITH_ANSWER_VG.json

# python examiners/dyna_conv.py --model_path llava-hf/llava-1.5-7b-hf --dataset coco --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_COCO_SIMGPLE_INTERROGATE --outfile output/vg/dyna/coco/llava-1.5-7b-hf/CONV_MODEL_PERSPECTIVE_PROMPT_COCO_SIMGPLE_INTERROGATE.json --num_samples 5
python examiners/dyna_conv.py --model_path llava-hf/llava-1.5-7b-hf --dataset coco --p_mode CONV_MODEL_PERSPECTIVE_PROMPT_COCO_UNANSWERABLE --outfile output/vg/dyna/coco/llava-1.5-7b-hf/CONV_MODEL_PERSPECTIVE_PROMPT_COCO_UNANSWERABLE.json --num_samples 5