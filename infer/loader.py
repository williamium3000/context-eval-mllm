from .infer_llava import eval_model as eval_model_llava
from .infer_blip2 import eval_model as eval_model_blip2
from .infer_qwenvl2d5 import eval_model as eval_model_qwenvl2d5
from functools import partial

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(args):
    if "llava-1.5" in args.model_path:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(0)

        processor = AutoProcessor.from_pretrained(args.model_path)
        return partial(eval_model_llava, model=model, processor=processor)
    
    elif "blip2" in args.model_path:
        processor = Blip2Processor.from_pretrained(args.model_path)
        model = Blip2ForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float32
        )
        model.to(device)
        return partial(eval_model_blip2, model=model, processor=processor)
    elif "Qwen2.5-VL" in args.model_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype="auto", device_map="auto"
        )

        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2.5-VL-3B-Instruct",
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map="auto",
        # )

        # default processer
        processor = AutoProcessor.from_pretrained(args.model_path)
        return partial(eval_model_qwenvl2d5, model=model, processor=processor)
    else:
        raise NotImplementedError