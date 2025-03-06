export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0

# python graders/chair/convert.py output/dyna_bad_examples/coverage_certainty.json
# python graders/chair/chair.py graders/chair/output/coverage_certainty.json

python graders/chair/chair_vg.py output/vg/dyna/CONV_MODEL_PERSPECTIVE_PROMPT_VG_SIMGPLE3.json filtered_object_synsets_final.json
