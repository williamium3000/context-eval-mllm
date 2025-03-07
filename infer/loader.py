from .infer_llava import eval_model
from functools import partial

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


def load_model(args):
    if "llava-1.5-7b-hf" in args.model_path:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(0)

        processor = AutoProcessor.from_pretrained(args.model_path)
        return partial(eval_model, model=model, processor=processor)
    else:
        raise NotImplementedError