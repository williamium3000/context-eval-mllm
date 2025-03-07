import torch
import os
import argparse
import json
import tqdm
from PIL import Image


import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"


def eval_model(processor, model, image_file, query):
    # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image") 
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": query},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    raw_image = Image.open(requests.get(image_file, stream=True).raw)
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    print(output)
    output = processor.decode(output[0][2:], skip_special_tokens=True)

    return output





