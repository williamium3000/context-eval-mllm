export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0

python grader/ha_dpo_grader/eval.py \
    work_dirs/vg/dyna-v6/llava-1.5-7b-hf.json \
    --outdir work_dirs/vg/dyna-v6/llava-1.5-7b-hf-ha_dpo_grader