from .coco import load_coco2017
from .vg import load_vg

import os
from PIL import Image
import argparse
import torch
import json
import requests
from PIL import Image
from io import BytesIO
import re
import os
import tqdm

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_data(args):
    if args.dataset == "coco":
        samples =  load_coco2017(args.num_samples)
        for sample in samples:
            sample["image"] = load_image(os.path.join("data/coco/val2017", sample["file_name"]))
            
    elif args.dataset == "vg":
        samples = load_vg(args.num_samples)
    else:
        raise ValueError("Unknown dataset")
    return samples

    