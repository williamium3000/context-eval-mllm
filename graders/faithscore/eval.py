import argparse
import json
import os
from tqdm import tqdm
from framework import FaithScore
from io import BytesIO
import requests

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conv', type=str, default='output/vg/caption.json')
    parser.add_argument('--vem_type', type=str, default="llava", choices=["ofa-ve", "ofa", "llava"])
    parser.add_argument('--llava_path', type=str, default="checkpoints/llava-v1.5-7b")
    parser.add_argument('--use_llama', action='store_true')
    parser.add_argument('--llama_path', type=str)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    score = FaithScore(vem_type=args.vem_type, api_key=api_key,
                       llava_path=args.llava_path, use_llama=args.use_llama,
                       llama_path=args.llama_path)

    images = []
    answers = []
    conv_data = json.load(open(args.conv, 'r'))
    for sample in tqdm(conv_data):
        for conv in sample["conversations"]:
            response = conv["response"].strip().replace('\n', '')
            image_url = sample["url"]

            try:
                request_response = requests.get(image_url)
                images.append(BytesIO(request_response.content))
                answers.append(response)
            except requests.exceptions.RequestException as e:
                print(f"Error retrieving image: {e}")
                continue

    score, sentence_score = score.faithscore(answers, images)

    print('Overall score:', score)
    print('Sentence score:', sentence_score)
