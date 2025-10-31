export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0

python grader/HaELM/haelm.py \
    --conv output/vg/caption/Qwen2.5-VL-3B-Instruct-test.json \
    --checkpoint_path /home/ubuntu/william/repo/LLaMA-Factory/saves/gpt-20b/lora/merged-lr5e-4 \
    --outfile output/vg/caption/Qwen2.5-VL-3B-Instruct-test_haelm.json