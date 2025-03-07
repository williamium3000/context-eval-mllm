from utils.utils import load_data
from infer.loader import load_model
import os
import argparse
import json
import tqdm
import copy

def dyna_conv(case, eval_func):
    message_evaluator = "Please provide a detailed description."
    image_file = case["image"]
    output = eval_func(image_file=image_file, query=message_evaluator)
    output = output.lower()
    to_save = [
        {"round_id": 0, "prompt": "Please provide a brief description.", "response": output}
    ]
    
    return to_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_samples', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    # need to figure out how to eval on different models
    eval_func = load_model(args)
    samples = load_data(args)
    
    print("starting conversation with model...")
    for sample in tqdm.tqdm(samples):
        conv = dyna_conv(sample, eval_func)
        sample["conversations"] = conv
        del sample["image"]
    
    
    with open(args.outfile, "w") as f:
        json.dump(samples, f, indent=4)
