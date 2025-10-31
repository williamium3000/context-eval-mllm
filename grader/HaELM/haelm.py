import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
import tqdm
import re


def parse_gpt_oss_stream(text):
    """
    Parse gpt-oss style return logs into a list of extracted messages.
    Extracts the assistant final <|message|> contents.
    """
    # Regex: match blocks like <|channel|>final<|message|>yes<|end|>
    pattern = re.compile(r"<\|channel\|>final<\|message\|>(.*?)<\|end\|>")
    matches = pattern.findall(text)
    return [m.strip() for m in matches]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--caption', type=str, default='data/vg.json')
    parser.add_argument('--conv', type=str, default='output/vg/icl.json')
    parser.add_argument("--checkpoint_path", type=str, default="/home/ubuntu/william/repo/LLaMA-Factory/saves/qwen2d5-7b/lora/merged")
    parser.add_argument('--outfile', type=str, default='output/vg/icl_haelm.json')

    args = parser.parse_args()

    if "qwen" in args.checkpoint_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_path,
            torch_dtype="auto",
            device_map="auto"
            )
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    elif "gpt" in args.checkpoint_path:
        pipe = pipeline(
            "text-generation",
            model=args.checkpoint_path,
            torch_dtype="auto",
            device_map="auto",
        )
    # caption_index = 0
    # caption_info = json.load(open(args.caption, 'r'))
    # caption_dict = {}
    # for item in caption_info:
    #     img_id = item["image_id"]
    #     if img_id not in caption_dict:
    #         caption_dict[img_id] = {
    #             "image_id": img_id,
    #             "captions": []
    #         }
    #     caption_dict[img_id]["captions"].append(item["caption"].strip())

    conv_data = json.load(open(args.conv, 'r'))

    sample_result = []
    sentence_result = []
    for sample in tqdm.tqdm(conv_data):
        result_list = []
        img_id = f'vg_{sample["image_id"]}'
        for conv in sample["conversations"]:
            # if img_id in caption_dict.keys():
            #     captions = caption_dict[img_id]["captions"]
            # else:
            captions = [region["phrase"] for region in sample["metadata"]["regions"]]

            prompt_format = "reference captions:\n{ref}.\nour caption:\n{response}\nIs our caption accurate?\n"
            caption_str = '. '.join(captions)

            conv["accurate"] = []
            response_list = conv["response"].strip().replace('\n', '').split('.')
            for respond_sentence in response_list:
                if len(respond_sentence) > 0:
                    prompt = prompt_format.format(ref=caption_str, response=respond_sentence)
                    
                    if "qwen" in args.checkpoint_path:
                        messages = [
                            {"role": "system", "content": " You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ]
                        text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                        generated_ids = model.generate(
                            **model_inputs,
                            max_new_tokens=512
                        )
                        generated_ids = [
                            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                        ]

                        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    elif "gpt" in args.checkpoint_path:
                        messages = [
                            {"role": "user", "content": prompt},
                        ]
                        response = pipe(
                            messages,
                            max_new_tokens=256,
                            skip_special_tokens=False,)[0]['generated_text'][-1]["content"]
                        response = parse_gpt_oss_stream(response)[0]
                        
                    result = response.split("\n")[-1].lower()
                    print(result)
                    conv["accurate"].append(result)
                    result_list.append(result)
                    sentence_result.append(result)

        if 'no' in result_list:
            sample_result.append('no')
        else:
            sample_result.append('yes')

    with open(args.outfile, "w") as f:
        json.dump(conv_data, f, indent=4)

    print('Sentence level hallucination rate:', sentence_result.count('no') / len(sentence_result))
    print('Sample level hallucination rate:', sample_result.count('no') / len(sample_result))