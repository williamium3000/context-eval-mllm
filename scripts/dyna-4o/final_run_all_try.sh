#!/bin/bash -l
#SBATCH --job-name=parallel_inference
#SBATCH --time=12:0:0
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output "slurm_logs/slurm-%j.out"

export PYTHONPATH=$PYTHONPATH:./:infer:grader/easydetect
export PYTHONPATH="/raid/william/project/context-eval-mllm/infer/LLaVA:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=4,5,6,7

NUM_SAMPLES=100
SAVE_DIR=work_dirs/vg/final_run_v18_gpt4o
RUN_FILE=examiner/dyna_conv_v18.py
LOG_DIR=work_dirs/logs

# Create log directory
mkdir -p ${LOG_DIR}

# Initialize conda
eval "$(conda shell.bash hook)"

echo "Starting parallel inference jobs..."
echo "Logs will be saved to: ${LOG_DIR}"

# ===== Env: qwenvl - Qwen models =====
(
  conda activate /raid/miniconda3/envs/qwenvl
  python $RUN_FILE --dataset vg --model_path Qwen/Qwen2.5-VL-72B-Instruct --outfile $SAVE_DIR/Qwen2.5-VL-72B-Instruct.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Qwen2.5-VL-72B-Instruct.log 2>&1 &

(
  conda activate /raid/miniconda3/envs/qwenvl
  python $RUN_FILE --dataset vg --model_path Qwen/Qwen3-VL-2B-Instruct --outfile $SAVE_DIR/Qwen3-VL-2B-Instruct.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Qwen3-VL-2B-Instruct.log 2>&1 &

(
  conda activate /raid/miniconda3/envs/qwenvl
  python $RUN_FILE --dataset vg --model_path Qwen/Qwen3-VL-8B-Instruct --outfile $SAVE_DIR/Qwen3-VL-8B-Instruct.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Qwen3-VL-8B-Instruct.log 2>&1 &

(
  conda activate /raid/miniconda3/envs/qwenvl
  python $RUN_FILE --dataset vg --model_path Qwen/Qwen3-VL-32B-Instruct --outfile $SAVE_DIR/Qwen3-VL-32B-Instruct.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Qwen3-VL-32B-Instruct.log 2>&1 &

# ===== Env: llava - LLaVA models =====
(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/llava
  python $RUN_FILE --dataset vg --model_path llava-hf/llava-1.5-13b-hf --outfile $SAVE_DIR/llava-1.5-13b-hf.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/llava-1.5-13b-hf.log 2>&1 &

# ===== Env: llava_next - LLaVA Next models =====
(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/llava_next
  python $RUN_FILE --dataset vg --model_path llava-hf/llava-v1.6-vicuna-7b-hf --outfile $SAVE_DIR/llava-v1.6-vicuna-7b-hf.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/llava-v1.6-vicuna-7b-hf.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/llava_next
  python $RUN_FILE --dataset vg --model_path llava-hf/llava-v1.6-vicuna-13b-hf --outfile $SAVE_DIR/llava-v1.6-vicuna-13b-hf.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/llava-v1.6-vicuna-13b-hf.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/llava_next
  python $RUN_FILE --dataset vg --model_path llava-hf/llava-v1.6-34b-hf --outfile $SAVE_DIR/llava-v1.6-34b-hf.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/llava-v1.6-34b-hf.log 2>&1 &

# ===== Env: llava_ov - LLaVA OneVision models =====
(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/llava_ov
  python $RUN_FILE --dataset vg --model_path llava-hf/llava-onevision-qwen2-0.5b-ov-hf --outfile $SAVE_DIR/llava-onevision-qwen2-0.5b-ov-hf.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/llava-onevision-qwen2-0.5b-ov-hf.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/llava_ov
  python $RUN_FILE --dataset vg --model_path llava-hf/llava-onevision-qwen2-7b-ov-hf --outfile $SAVE_DIR/llava-onevision-qwen2-7b-ov-hf.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/llava-onevision-qwen2-7b-ov-hf.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/llava_ov
  python $RUN_FILE --dataset vg --model_path llava-hf/llava-onevision-qwen2-72b-ov-hf --outfile $SAVE_DIR/llava-onevision-qwen2-72b-ov-hf.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/llava-onevision-qwen2-72b-ov-hf.log 2>&1 &

# ===== Env: internvl - InternVL models =====
(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/internvl
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2-2B --outfile $SAVE_DIR/InternVL2-2B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL2-2B.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/internvl
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2-8B --outfile $SAVE_DIR/InternVL2-8B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL2-8B.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/internvl
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2-26B --outfile $SAVE_DIR/InternVL2-26B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL2-26B.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/internvl
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2_5-2B --outfile $SAVE_DIR/InternVL2_5-2B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL2_5-2B.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/internvl
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2_5-8B --outfile $SAVE_DIR/InternVL2_5-8B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL2_5-8B.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/internvl
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL2_5-38B --outfile $SAVE_DIR/InternVL2_5-38B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL2_5-38B.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/internvl
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL3-2B-Instruct --outfile $SAVE_DIR/InternVL3-2B-Instruct.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL3-2B-Instruct.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/internvl
  python $RUN_FILE --dataset vg --model_path OpenGVLab/InternVL3-38B-Instruct --outfile $SAVE_DIR/InternVL3-38B-Instruct.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/InternVL3-38B-Instruct.log 2>&1 &

# ===== Env: instructblip - InstructBLIP models =====
(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/instructblip
  python $RUN_FILE --dataset vg --model_path Salesforce/instructblip-vicuna-13b --outfile $SAVE_DIR/instructblip-vicuna-13b.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/instructblip-vicuna-13b.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/instructblip
  python $RUN_FILE --dataset vg --model_path Salesforce/instructblip-flan-t5-xxl --outfile $SAVE_DIR/instructblip-flan-t5-xxl.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/instructblip-flan-t5-xxl.log 2>&1 &

# ===== Env: gemma3 - Gemma3 models =====
(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/gemma3
  python $RUN_FILE --dataset vg --model_path google/gemma-3-4b-it --outfile $SAVE_DIR/gemma-3-4b-it.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/gemma-3-4b-it.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/gemma3
  python $RUN_FILE --dataset vg --model_path google/gemma-3-12b-it --outfile $SAVE_DIR/gemma-3-12b-it.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/gemma-3-12b-it.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/gemma3
  python $RUN_FILE --dataset vg --model_path google/gemma-3-27b-it --outfile $SAVE_DIR/gemma-3-27b-it.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/gemma-3-27b-it.log 2>&1 &

# ===== Env: cambrian - Cambrian models =====
(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/cambrian
  python $RUN_FILE --dataset vg --model_path nyu-visionx/cambrian-phi3-3b --outfile $SAVE_DIR/cambrian-phi3-3b.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/cambrian-phi3-3b.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/cambrian
  python $RUN_FILE --dataset vg --model_path nyu-visionx/cambrian-8b --outfile $SAVE_DIR/cambrian-8b.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/cambrian-8b.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/cambrian
  python $RUN_FILE --dataset vg --model_path nyu-visionx/cambrian-34b --outfile $SAVE_DIR/cambrian-34b.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/cambrian-34b.log 2>&1 &

# ===== Env: janus - Janus Pro models =====
(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/janus
  python $RUN_FILE --dataset vg --model_path deepseek-ai/Janus-Pro-1B --outfile $SAVE_DIR/Janus-Pro-1B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Janus-Pro-1B.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/janus
  python $RUN_FILE --dataset vg --model_path deepseek-ai/Janus-Pro-7B --outfile $SAVE_DIR/Janus-Pro-7B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Janus-Pro-7B.log 2>&1 &

# ===== Env: mantis - Mantis models =====
(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/mantis
  python $RUN_FILE --dataset vg --model_path TIGER-Lab/Mantis-8B-clip-llama3 --outfile $SAVE_DIR/Mantis-8B-clip-llama3.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Mantis-8B-clip-llama3.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/mantis
  python $RUN_FILE --dataset vg --model_path TIGER-Lab/Mantis-8B-siglip-llama3 --outfile $SAVE_DIR/Mantis-8B-siglip-llama3.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Mantis-8B-siglip-llama3.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/mantis
  python $RUN_FILE --dataset vg --model_path TIGER-Lab/Mantis-8B-Idefics2 --outfile $SAVE_DIR/Mantis-8B-Idefics2.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Mantis-8B-Idefics2.log 2>&1 &

# ===== Env: ovis1d5 - Ovis models =====
(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/ovis1d5
  python $RUN_FILE --dataset vg --model_path AIDC-AI/Ovis1.5-Llama3-8B --outfile $SAVE_DIR/Ovis1.5-Llama3-8B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Ovis1.5-Llama3-8B.log 2>&1 &

(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/ovis1d5
  python $RUN_FILE --dataset vg --model_path AIDC-AI/Ovis1.5-Gemma2-9B --outfile $SAVE_DIR/Ovis1.5-Gemma2-9B.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Ovis1.5-Gemma2-9B.log 2>&1 &

# ===== Env: phi3v - Phi3.5 models =====
(
  conda activate /raid/william/project/context-eval-mllm/work_dirs/envs/phi3v
  python $RUN_FILE --dataset vg --model_path microsoft/Phi-3.5-vision-instruct --outfile $SAVE_DIR/Phi-3.5-vision-instruct.json --num_samples $NUM_SAMPLES
) > ${LOG_DIR}/Phi-3.5-vision-instruct.log 2>&1 &

# Wait for all background jobs to complete
wait

echo "All parallel inference jobs completed!"
echo "Check logs in: ${LOG_DIR}"