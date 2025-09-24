export PYTHONPATH=$PYTHONPATH:./:infer
export CUDA_VISIBLE_DEVICES=6

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate work_dirs/envs/qwenvl2d5

python examiner/dyna_conv_v2.py \
    --dataset vg --model_path llava-hf/llava-1.5-7b-hf  \
    --outfile work_dirs/vg/dyna-v2-context-regular/llava-1.5-7b-hf.json \
    --num_samples 5
