export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0

# python graders/chair/convert.py output/dyna_bad_examples/coverage_certainty.json
# python graders/chair/chair.py graders/chair/output/coverage_certainty.json

python graders/chair_vg/convert.py output/vg/dyna/icl.json
python graders/chair_vg/chair.py graders/chair/output/icl.json object_synsets.json
