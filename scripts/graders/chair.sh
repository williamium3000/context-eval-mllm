export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0

# python graders/chair/convert.py output/dyna_bad_examples/coverage_certainty.json
# python graders/chair/chair.py graders/chair/output/coverage_certainty.json

python graders/chair/chair_vg.py $1 filtered_object_synsets_final.json
