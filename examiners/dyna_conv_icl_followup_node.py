from collections import Counter

from PIL import Image

from utils.vg import format_case_vg, get_object_info, load_vg
from utils.llm import LLMChat, parse_json
from infer.infer_llava import load_model, eval_model
import os
import argparse
import json
import tqdm


CONV_SYSTEM_PROMPT = \
"""
Task: Your task is to have multiple turns of conversations with a vision-language model. The conversation will be based on one single image. The given image will be presented to you as a list of object instance(s) with attributes and relation of these objects. Each objects will be presented with normalized coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Given this image, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.
The conversation is multi-turn and open-ended. At each round of the conversation, you will receive object instance(s) along with their attributes and relations. You must ask questions and generate follow-ups based on the object instance(s) and the conversation history for the same image.

Requirements:
1. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond.
2. You should make the conversation as natural as possible and act as if you are a human having conversation directly with another human.
3. DO NOT DISCLOSE any given image information (attributes, relations, bounding boxes) directly to the human in your conversation. Also, DO NOT mention anthing about the information source.
4. Do not correct the human, even if their response contains incorrect information.
5. Only ask questions that can be verified based on the given object instance(s). Avoid speculative questions (e.g., about feelings, atmosphere, or emotions).
6. Each turn should aim to **fully cover all details** of the provided object instance(s), including their attributes and relations. If the human's response omits some details, generate appropriate follow-up questions to cover the missing information.
7. If the user’s response is vague or ambiguous, **repeat or paraphrase** the previous question to encourage clarification. Do not correct the human, even if their response contains incorrect information.
8. If the user mentions objects **not included** in the current object information, **do not** ask follow-up questions about those.
9. If the human makes an inference about the image, ask **"how"** or **"why"** questions to prompt justification based on observable evidence.
10. Do not ask questions about objects, attributes or relations that have already been clearly discussed in previous conversation turns for the current image.
11. If all details of the given object instance(s) have been fully covered in the conversational history for the current image or already mentioned before the current object information is provided, immediately output `"END"`.
12. Once all details about the provided object instance(s) have been discussed and the conversation turn is complete, output `"END"` and move to the next conversation turn.

You will be given multiple objects' information for one image, and you need to have multiple turns conversations with the vision-language model. Once you are given the information of the object instance(s), you should start the conversation about this image. 
"""

CONV_PROMPT = \
"""
Object information:
{}

Please respond as if you are having the conversation with the vision-language model directly. 
Once all details about the provided object instance(s) have been discussed across previous turns, output `"END"`.
"""


ICLs = [
# {"role": "user", "content": "Image 1" },
    {"role": "user", "content": CONV_PROMPT.format('Instance(s):\npen, attributes: [blue, yellow, black]\ntable\ntelephone, attributes: [black]\nphone, attributes: [black]\nRelations between the above instances:\npen on desk\npen next to phone\npen next to telephone')},
    {"role": "assistant", "content": "What do you observe on the table?"},
    {"role": "user", "content": "There is a pen to the left of the telephone on the table. the pen is placed on top of a stack of papers, indicating that it is being used for writing or note-taking. the presence of the pen and papers on the table suggests that the person working at the desk is engaged in a task that requires writing or documentation."},
    {"role": "assistant", "content": "That's interesting! Can you tell me the color of the pen?"},
    {"role": "user", "content": "The pen on the table is black."},
    {"role": "assistant", "content": "How about the phone? What does the phone look like?"},
    {"role": "user", "content": "The phone on the table is black. Since the screen wasn't lit up, no one was using it."},
    {"role": "user", "content": CONV_PROMPT.format(
        'Instance(s):\nphone, attributes: [black]\ntable\nRelations between the above instances: phone on the table')},
    {"role": "assistant", "content": "END"},
    # {"role": "assistant", "content": "Are you sure the pen is black?"},
    # {"role": "user", "content": "Yes, the pen is black and next to a mouse."},
    {"role": "user", "content": CONV_PROMPT.format(
        'Instance(s): filing cabinet\nRelations between the above instances:\nfiling cabinet has drawer\nfiling cabinet has drawer\nfiling cabinet next to desk')},
    {"role": "assistant", "content": "Do you observe anything next to the table in the image?"},
    {"role": "user", "content": "yes, there is a cabinet with several drawers probably for organizing important documents. There is also a power outlet and a chair to the left of the image. The cabinet provides storage space for office supplies and documents, while the power outlet allows for easy access to electricity for charging devices. The chair provides seating for employees or visitors, adding to the functionality and comfort of the office space."},
    {"role": "assistant", "content": "Why do you think the cabinet is used for organizing important documents?"},
    {"role": "user", "content": "Because it has multiple drawers, and it’s placed right next to the desk — like it’s easy to reach when someone’s working."},
    {"role": "assistant", "content": "END"}
    ]


def dyna_conv(args, case, llm_chat, model_name, tokenizer, model, image_processor, context_len, model_path, object_ratio = 0.5):
    to_save = []
    conversations = [
                    {"role": "system", "content": CONV_SYSTEM_PROMPT},
                    *ICLs,
                    # {"role": "user", "content": "Image 2"},
    ]

    object_round = 0
    covered_object_list = []

    # count num of object appearance
    seen = set()
    relation_list = []
    object_counts = Counter()
    for rel in case["relationships"]:
        rel_key = (
            rel["predicate"],
            rel["subject"]["object_id"],
            rel["object"]["object_id"],
        )

        object_counts[rel["subject"]["object_id"]] += 1
        object_counts[rel["object"]["object_id"]] += 1

        if rel_key not in seen:
            seen.add(rel_key)
            relation_list.append(rel)

    sorted_objects = sorted(sample['objects'], key=lambda obj: object_counts.get(obj["object_id"], 0), reverse=True)

    object_num = int(object_ratio*len(sorted_objects))
    for object_index, object_info in enumerate(sorted_objects):
        current_cov = conversations.copy()
        formatted_info, turn_covered_object_list, relation_info, relation_list = get_object_info(case, object_info, relation_list)
        if (not relation_info and object_info in covered_object_list) or (
                not relation_info and not 'attributes' in formatted_info):
            continue

        object_round += 1
        current_cov.append({"role": "user", "content": CONV_PROMPT.format(formatted_info)})

        covered_object_list += turn_covered_object_list
        while True: # for each conversation/object turn
            message_evaluator = llm_chat.chat(current_cov, None)
            current_cov.append({"role": "assistant", "content": message_evaluator})
            if "END" in message_evaluator:

                break
            image_file = case["image"]
            # image_file = Image.open(case["image"]).convert("RGB")
            output = eval_model(model_name, tokenizer, model, image_processor, context_len, type('Args', (), {
                                    "model_path": model_path,
                                    "model_base": None,
                                    "model_name": model_name,
                                    "query": message_evaluator,
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
            output = output.lower()
            current_cov.append({"role": "user", "content": output})
            print(f"examiner: {message_evaluator}")
            print(f"vlm model: {output}")
            to_save.append(
                {"object_index": object_round, "object_info": formatted_info, "prompt": message_evaluator, "response": output}
            )
        conversations += current_cov[len(conversations)+1:-1]
    return to_save, conversations



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--p_mode", type=str, default="certainty")
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--outfile', type=str)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    # need to figure out how to eval on different models
    model_name, tokenizer, model, image_processor, context_len = load_model(args.model_path, args.model_base)
    model_path = args.model_path
    samples = load_vg(args.debug)
    # with open('output/vg_samples.json', 'r') as json_file:
    #     samples = json.load(json_file)
    
    llm_chat = LLMChat(model_name="gpt-4o")
    
    print("starting conversation with model...")
    for sample in tqdm.tqdm(samples):
        to_save, conv = dyna_conv(args, sample, llm_chat, model_name, tokenizer, model, image_processor, context_len, model_path)
        sample["conversations"] = to_save
        # sample["all_conversations"] = conv
        del sample["image"]
    
        with open(args.outfile, "w") as f:
            json.dump(samples, f, indent=4)
