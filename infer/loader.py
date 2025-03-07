from .infer_llava import eval_model as eval_model_llava
from .infer_blip2 import eval_model as eval_model_blip2
from functools import partial

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration

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
    
    
    else:
        raise NotImplementedError