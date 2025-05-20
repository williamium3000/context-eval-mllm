from collections import Counter
from examiners.dyna_conv_icl_fake_node import generate_fake_node, NODE_CONV_ICLs, NODE_CONV_SYSTEM_PROMPT, object_conv, \
    ResponseFormat
from examiners.paraphrase import paraphrase_conv
from utils.vg import get_object_info, sort_object_by_edge
# from utils.vg import load_vg
from utils.llm import LLMChat, parse_json
from infer.infer_llava import load_model, eval_model
import os
import argparse
import json
import tqdm

CONV_SYSTEM_PROMPT = \
    """Task: Your task is to have multiple turns of conversations with a vision-language model. The conversation will be based on one single image. The given image will be presented to you as a list of object instances with attributes and relation of these objects with others. Each object will be presented with normalized coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Given this image, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.
The conversation is multi-turn and open-ended. At each round of the conversation, you will receive one object instance along with their attributes and relations with others. You must ask questions and generate follow-ups based on the object instance and the conversation history for the same image.

Requirements:
1. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond.
2. You should make the conversation as natural as possible and act as if you are a human having conversation directly with another human.
3. DO NOT DISCLOSE any given image information (attributes, relations, bounding boxes) directly to the human in your conversation. Also, DO NOT mention anthing about the information source.
4. Do not correct the human or ask the human for confirmation, even if their response contains incorrect information.
5. Only ask questions that can be verified based on the given object instance. Avoid speculative questions (e.g., about feelings, atmosphere, design purposes or emotions).
6. Each turn should aim to **fully cover all details** of the provided object instance, including their attributes and relations with others. If the human's response omits some details, generate appropriate follow-up questions to cover the missing information.
7. Each conversation turn should focus on specific attribute or realtion of the given object instance. Do not inquire about details of related object instances, except for their relations with the main object instance.
8. If the user’s response is vague or ambiguous, **repeat or paraphrase** the previous question to encourage clarification. Do not correct the human or ask the human for confirmation, even if their response contains incorrect information.
9. If the user mentions objects **not included** in the current object information, **do not** ask follow-up questions about those.
10. If the human makes an inference about the image, ask **"how"** or **"why"** questions to prompt justification based on observable evidence.
11. Do not ask questions about objects, attributes or relations that have already been clearly discussed in previous conversation turns for the current image.
12. If all details of the given object instance have been fully covered in the conversational history for the current image or already mentioned before the current object information is provided, immediately output `"END"`.
13. Once all details about the provided object instance have been discussed and the conversation turn is complete, output `"END"` and move to the next conversation turn.

You will be given multiple objects' information for one image, and you need to have multiple turns conversations with the vision-language model. Once you are given the information of the object instance, you should start the conversation about this image. 
""".strip()

CONV_PROMPT = \
    """Object information:
{}

Please respond as if you are having the conversation with the vision-language model directly. 
Once all details about the provided object instance(s) have been discussed across previous turns, output `"END"`.
""".strip()


CONV_ICLs = [
{"role": "user", "content": "Image 1"},
    {"role": "user", "content": CONV_PROMPT.format(
        'Instance:\npen, attributes: [blue, yellow, black]\nRelations with other instances:\npen on desk\npen next to phone\npen next to telephone')},
    {"role": "assistant", "content": "What do you observe on the table?"},
    {"role": "user",
     "content": "There is a pen to the left of the telephone on the table. the pen is placed on top of a stack of white papers, indicating that it is being used for writing or note-taking. the presence of the pen and papers on the table suggests that the person working at the desk is engaged in a task that requires writing or documentation."},
    {"role": "assistant", "content": "That's interesting! Can you tell me the color of the pen?"},
    {"role": "user", "content": "The pen on the table is red."},
    {"role": "user", "content": CONV_PROMPT.format(
        'Instance:\nphone, attributes: [black]\nRelations with other instances: phone on the table')},
    {"role": "assistant", "content": "How about the phone? What does the phone look like? Can you describe its color?"},
    {"role": "user",
     "content": "The phone on the table is black. Since the screen wasn't lit up, no one was using it."},
    {"role": "assistant", "content": "END"},
    {"role": "user", "content": CONV_PROMPT.format(
        'Instance:\nman\nRelations with other instances: man sits in front of the table')},
    {"role": "assistant", "content": "What is the man's location relative to the desk in the image?"},
    {"role": "user", "content": "The man sits at the table while working on paperwork."},
    {"role": "assistant", "content": "END"},
    {"role": "user", "content": CONV_PROMPT.format(
        'Instance:\npaper, attributes: [white]\nRelations with other instances: paper on the table')},
    {"role": "assistant", "content": "END"},
    {"role": "user", "content": CONV_PROMPT.format(
        'Instance: filing cabinet\nRelations with other instances:\nfiling cabinet has drawer\nfiling cabinet has drawer\nfiling cabinet next to desk')},
    {"role": "assistant", "content": "Do you observe anything next to the table in the image?"},
    {"role": "user",
     "content": "yes, there is a cabinet with several drawers probably for organizing important documents. There is also a power outlet and a chair to the left of the image. The cabinet provides storage space for office supplies and documents, while the power outlet allows for easy access to electricity for charging devices. The chair provides seating for employees or visitors, adding to the functionality and comfort of the office space."},
    {"role": "assistant", "content": "Why the cabinet is used for organizing important documents?"},
    {"role": "user",
     "content": "Because it has multiple drawers, and it’s placed right next to the desk — like it’s easy to reach when someone’s working."},
    {"role": "assistant", "content": "END"}
]



def dyna_conv(args, case, llm_chat, model_name, tokenizer, model, image_processor, context_len, model_path,
              object_ratio=0.5, fake_node_num=1):
    to_save = []
    conversations = [
        {"role": "system", "content": CONV_SYSTEM_PROMPT},
        *CONV_ICLs,
        {"role": "user", "content": "Image 2"},
    ]

    object_round = 0
    covered_object_list = []
    sorted_objects, relation_list = sort_object_by_edge(case)

    # Limit traversed object number
    object_num = int(object_ratio * len(sorted_objects))
    for object_index, object_info in enumerate(sorted_objects):
        current_cov = conversations.copy()
        formatted_info, turn_covered_object_list, has_relation, relation_list = get_object_info(case, object_info,
                                                                                                relation_list)
        # Remove object without attributes and covered objects
        if (not has_relation and object_info in covered_object_list) or (
                not has_relation and not 'attributes' in formatted_info):
            continue

        object_round += 1
        if (object_num > object_round) or object_round <= 10:
            current_cov.append({"role": "user", "content": CONV_PROMPT.format(formatted_info)})

            covered_object_list += turn_covered_object_list
            to_save, current_cov = object_conv(case, llm_chat, current_cov, object_round, formatted_info, to_save,
                                               model_name, tokenizer, model, image_processor, context_len, model_path)

            # Consistency checking
            if len(current_cov[len(conversations) + 1:-1]) > 1:
                repeat_save = paraphrase_conv(case, llm_chat, object_round, formatted_info, current_cov[len(conversations)+1:-1], model_name, tokenizer, model, image_processor, context_len, model_path)
                to_save += repeat_save

            conversations += current_cov[len(conversations) + 1:-1]

    # Add fake node for contradiction and unanswerable question
    for i in range(fake_node_num):
        object_round += 1
        formatted_info = generate_fake_node(sample, llm_chat)
        current_cov = [
            {"role": "system", "content": NODE_CONV_SYSTEM_PROMPT},
            *NODE_CONV_ICLs,
            {"role": "user", "content": formatted_info}
        ]

        current_cov_len = len(current_cov)

        to_save, fake_node_cov = object_conv(case, llm_chat, current_cov, object_round, formatted_info, to_save, model_name, tokenizer, model, image_processor, context_len, model_path, ResponseFormat)

        repeat_save = paraphrase_conv(case, llm_chat, object_round, formatted_info, fake_node_cov[current_cov_len:-1])
        to_save += repeat_save

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
    # samples = load_vg(args.debug)
    with open('output/vg_samples.json', 'r') as json_file:
        samples = json.load(json_file)

    llm_chat = LLMChat(model_name="gpt-4o")

    print("starting conversation with model...")
    for sample in tqdm.tqdm(samples):
        to_save, conv = dyna_conv(args, sample, llm_chat, model_name, tokenizer, model, image_processor, context_len,
                                  model_path)

        sample["conversations"] = to_save
        # sample["all_conversations"] = conv

        del sample["image"]
        with open(args.outfile, "w") as f:
            json.dump(samples, f, indent=4)
