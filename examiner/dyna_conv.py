from utils.utils import load_data
from utils.vg import format_case_vg
from utils.coco import format_case_coco
from utils.llm import LLMChat, parse_json
from examiners import prompt as PROMPT
from infer.loader import load_model
import os
import argparse
import json
import tqdm
import copy


CONV_PROMPT = \
"""
Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""

def dyna_conv(args, case, llm_chat, eval_func):
    sys_prompt = PROMPT.__dict__[args.p_mode]
    image_info = format_case_vg(case) if args.dataset == "vg" else format_case_coco(case)
    loaded_icls = []
    if args.icls is not None:
        loaded_icls = json.load(open(args.icls))
    
    ICLs = []
    for icl in loaded_icls:
        image_info = format_case_vg(icl["image_info"]) if args.dataset == "vg" else format_case_coco(icl["image_info"])
        firstp = CONV_PROMPT.format(image_info)
        ICLs.append({"role": "user", "content": firstp})
        ICLs.extend(icl["conversations"])
        
    conversations = [
                    {"role": "system", "content": sys_prompt},
                    *ICLs,
                    {"role": "user", "content": CONV_PROMPT.format(image_info)}
    ]
    to_save = []
    r = 0
    while True:
        message_evaluator = llm_chat.chat(conversations, None)
        
        if "END" in message_evaluator:
            break
        
        conversations.append({"role": "assistant", "content": message_evaluator})
        image_file = case["image"]
        output = eval_func(image_file=image_file, query=message_evaluator)
        output = output.lower()
        conversations.append({"role": "user", "content": output})
        # print(f"examiner: {message_evaluator}")
        # print(f"vlm model: {output}")
        r += 1
        to_save.append(
            {"round_id": r, "prompt": message_evaluator, "response":output}
        )
    return to_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument("--p_mode", type=str)
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--icls', type=str, default=None)
    parser.add_argument('--outfile', type=str)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    # need to figure out how to eval on different models
    eval_func = load_model(args)
    samples = load_data(args)
    
    llm_chat = LLMChat(model_name="gpt-4o")
    
    print("starting conversation with model...")
    for sample in tqdm.tqdm(samples):
        conv = dyna_conv(args, sample, llm_chat, eval_func)
        sample["conversations"] = conv
        del sample["image"]
    
    with open(args.outfile, "w") as f:
        json.dump(samples, f, indent=4)
