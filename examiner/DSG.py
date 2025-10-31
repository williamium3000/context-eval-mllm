import json
import tqdm
import os
from PIL import Image
import argparse
from openai import AzureOpenAI, OpenAI
from utils.query_utils import generate_dsg
from utils.parse_utils import parse_tuple_output, parse_question_output
from infer.infer_llava import load_model, eval_model
from functools import partial


if os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

if os.getenv("AZURE_OPENAI_KEY"):
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2025-01-01-preview")


def get_llm_response(
        prompt,
        model='gpt-4o-mini',
        temperature=0,
        return_response=False,
        max_tokens=500,
):
    if os.getenv("AZURE_OPENAI_KEY"):
        model = os.getenv("AZURE_OPENAI_DEPLOYNAME", model)

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if return_response:
        return completion

    return completion.choices[0].message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conv_script', type=str, default="output/caption/llava-1.5-7b-hf.json")
    parser.add_argument('--outfile', type=str, default="output/caption/pope/llava-1.5-7b-hf_pope.json")
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--pope_model_name', type=str, default='gpt-4o-mini')
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--sample_num', type=int, default=100)

    args = parser.parse_args()

    samples = json.load(open(args.conv_script, "r"))

    model_name, tokenizer, model, image_processor, context_len = load_model(args.model_path, args.model_base)
    model_path = args.model_path
    get_response = partial(get_llm_response, model=args.pope_model_name)
    for sample in tqdm.tqdm(samples[:args.sample_num]):
        conversation = sample["conversations"]

        id2prompts = {}
        for index, turn in enumerate(conversation):
            vlm_response = turn["response"]
            id2prompts[f'custom_{index}'] = {'input': vlm_response}

        id2tuple_outputs, id2question_outputs = generate_dsg(
            id2prompts,
            generate_fn=get_response,
            verbose=args.verbose
        )

        total_score = []
        for index, turn in enumerate(conversation):
            qid2tuple = parse_tuple_output(id2tuple_outputs[f'custom_{index}']['output'])
            qid2question = parse_question_output(id2question_outputs[f'custom_{index}']['output'])

            # qid2answer = {}
            # qid2scores = {}
            pope_qa = []
            for id, question in qid2question.items():
                # image_file = sample["image"]
                image_file = Image.open(sample["image"]).convert("RGB")

                answer = eval_model(model_name, tokenizer, model, image_processor, context_len, type('Args', (), {
                    "model_path": model_path,
                    "model_base": None,
                    "model_name": model_name,
                    "query": question,
                    "conv_mode": None,
                    "image_file": image_file,
                    "sep": ",",
                    "load_in_8bit": False,
                    "load_in_4bit": False,
                    "temperature": 0.0,  # set as 0.0 for reproceduce
                    "top_p": None,
                    "num_beams": 1,
                    "max_new_tokens": 512
                })())
                pope_qa.append({"index": id, "question": question, "answer": answer})
                # qid2answer[id] = answer
                # qid2scores[id] = float('yes' in answer.lower())

            # average_score = sum(qid2scores.values()) / len(qid2scores)
            # total_score += qid2scores.values()
            # print("average score", average_score)
            # turn["pope_score"] = average_score

            # for id in qid2question:
            #     print("ID", id)
            #     print("question", qid2question[id])
            #     print("answer", qid2answer[id])
            #     print("score", qid2scores[id])
            #     print()

        # sample["pope_score"] = sum(total_score) / len(total_score)

    with open(args.outfile, "w") as f:
        json.dump(samples, f, indent=4)