
import os
import argparse
import json
import tqdm
import copy

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def dyna_conv(case):
    message_evaluator = "Please provide a detailed description."
    image_file = case["image"]
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_file,
                },
                {"type": "text", "text": message_evaluator},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    output = output.lower()
    to_save = [
        {"round_id": 0, "prompt": "Please provide a brief description.", "response": output}
    ]
    
    return to_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_samples', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    samples = load_data(args)
    
    print("starting conversation with model...")
    for sample in tqdm.tqdm(samples):
        conv = dyna_conv(sample)
        sample["conversations"] = conv
        del sample["image"]
    
    
    with open(args.outfile, "w") as f:
        json.dump(samples, f, indent=4)
