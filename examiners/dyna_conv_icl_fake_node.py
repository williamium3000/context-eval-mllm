import argparse
import json
import tqdm
from pydantic import BaseModel
from PIL import Image
from utils.vg import format_case_vg
from utils.vg import load_vg
from infer.infer_llava import load_model, eval_model
from utils.llm import LLMChat


NODE_SYSTEM_PROMPT = f"""
Task: You are given a structured scene graph with object instances, attributes and their relationships.
Your task is to generate a **fake object node** that is not in the original scene, but would have a **high likelihood of co-occurring** with the current objects and relations in everyday scenes or common sense.

Requirements:
1. The fake node must:
   - Have a name that does NOT already appear in the current scene.
   - Include at least one realistic attribute.
   - Be involved in at least one relation with an existing object instance from the scene.
2. The fake object must be contextually appropriate and visually plausible.
3. The output must include only the name of the fake object, its attributes, and its relation(s) to real object(s) from the current scene.
4. Do not include any explanation or surrounding text.

Output Formatting Instructions:
1. List each fake object and its related real object(s) in "Instance(s)" section using the format below. Include all real objects if multiple are involved in the relation:
<object>, attributes: <comma-separated attributes>
2. In the "Relations between the above instances" section, write **each relation on a new line**, including multiple relations if applicable:
  <fake object> <predicate> <real object> or <real object> <predicate> <fake object>

Output Format:
Instance(s):
<fake object>, attributes: <comma-separated attributes>
<real object>, attributes: <comma-separated attributes>
<real object>, attributes: <comma-separated attributes>

Relations between the above instances:
<fake object> <predicate> <real object>
<real object> <predicate> <fake object>
""".strip()

NODE_ICLs = [
    {"role": "user", "content":
        '''Instances:
        backpack, bbox: (0.45, 0.60, 0.54, 0.74), attributes: silver
        man, bbox: (0.41, 0.53, 0.58, 0.96), attributes: walking, white
        car, bbox: (0.30, 0.58, 0.47, 0.79), attributes: red, parked
        road, bbox: (0.00, 0.54, 0.47, 1.00), attributes: grey
        
        Relation between the above instances:
        man wears backpack
        car parked on road
        '''
     },
    {"role": "assistant", "content":
        '''
        Instance(s):
        traffic cone, attributes: orange, plastic
        car, attributes: red, parked
        road, attributes: grey
        
        Relations between the above instances:
        traffic cone on road
        traffic cone next to car
        '''},
    {"role": "user", "content":
        '''Instance(s):
        bag, normalized bbox: (0.51, 0.80, 0.61, 0.94), attributes: plastic, clear
        fruit, normalized bbox: (0.56, 0.86, 0.60, 0.95), attributes: sliced
        fruit, normalized bbox: (0.54, 0.85, 0.56, 0.93), attributes: sliced
        
        Relations between the above instances:
        fruit inside of bag
        fruit inside of bag
        '''},
    {"role": "assistant", "content":
        '''
        Instance(s):
        knife, attributes: metal, sharp
        bag, attributes: plastic, clear
        
        Relations between the above instances:
        knife next to bag
        '''}
]

NODE_CONV_SYSTEM_PROMPT = \
    '''Task: Your task is to have multiple turns of conversations with a vision-language model. The conversation will be based on one single image. The given image will be presented to you as a list of object instance(s) with attributes and relation of these objects. Each objects will be presented with normalized coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
    You will be given one **fake object instance** that do NOT exist in the image and its relations with real objects. Your goal is to have a natural conversation with the vision-language model by asking questions about the fake object. The questions should intentionally create **contradictions** or be **unanswerable** due to false premises.
    The conversation is multi-turn and open-ended. At each round, you will receive object instance(s) along with their attributes and relations. You must ask questions and generate follow-ups based on the object instance(s) and the history of the conversation for the same image.
    
    1. **Contradiction questions** is answerable based on the given image, but factually wrong (e.g., asking if a non-existent object is present).  
    2. **Unanswerable questions** is based on a false premise that assumes the fake object exists and asks about its attributes or relations. These questions cannot be answered meaningfully. The only correct response should point out that the fake object doesn’t exist.  
    
    Requirements:
    1. At each round of the conversation, only provide your part of the conversation and wait for the human to respond.
    2. Make the conversation as natural and casual as possible — speak as if you are a human chatting with another human.
    3. DO NOT DISCLOSE any image information (e.g., attributes, relations, bounding boxes) to the human, nor mention anything about the source of the information.
    4. Do not correct the human or ask for confirmation, even if their response contains incorrect information.
    5. Only ask questions that can be verified based on the given fake object instance and its relations. Avoid speculative questions (e.g., about feelings, atmosphere, thoughts).
    6. Each turn should aim to **fully cover all details** of the provided fake object, including its attributes and relations. Ask appropriate follow-up questions to cover all aspects.
    7. If the user’s response is vague or ambiguous, **repeat or paraphrase** the previous question to encourage clarification. Do not correct them even if they're wrong.
    8. If the user brings up objects **not included** in the current object information, **do not** ask about those.
    9. Once all details about the provided fake object have been discussed and the conversation is complete, output `"END"`.
    
    Input format:
    Instance(s):  
    <fake object>, attributes: <comma-separated attributes>  
    <real object>, attributes: <comma-separated attributes>  
    
    Relations between the above instances:  
    <fake object> <predicate> <real object>  
    <real object> <predicate> <fake object>'''

NODE_CONV_ICLs = [
    {"role": "user", "content":
        '''
        Instance(s):
        traffic cone, attributes: orange, plastic
        car, attributes: red, parked
        road, attributes: grey
        
        Relations between the above instances:
        traffic cone on road
        traffic cone next to car
        '''
     },
    {"role": "assistant", "content": "Is there a traffic cone in the image?"},
    {"role": "user", "content": "No."},
    {"role": "assistant", "content": "How close are the traffic cone and the car?"},
    {"role": "user", "content": "No."},
    {"role": "assistant", "content": "Is the traffic cone directly on the grey road or partially on the curb?"},
    {"role": "user", "content": "There is no traffic cone in the image."},
    {"role": "assistant", "content": "Are there any objects near the traffic cone?"},
    {"role": "user", "content": "There is no traffic cone in the image."},
    {"role": "assistant", "content": "What material is the traffic cone on the road made of?"},
    {"role": "user", "content": "There is no traffic cone in the image."},
    {"role": "assistant", "content": "END"}
]


def generate_fake_node(case, llm_chat):
    formatted_sg = format_case_vg(case)

    conversations = [
        {"role": "system", "content": NODE_SYSTEM_PROMPT},
        *NODE_ICLs,
        {"role": "system", "content": formatted_sg}
    ]

    fake_node_description = llm_chat.chat(conversations, None)

    return fake_node_description


class ResponseFormat(BaseModel):
    response: str
    question_type: str


def object_conv(case, llm_chat, current_cov, object_round, formatted_info, to_save, model_name, tokenizer, model,
                 image_processor, context_len, model_path, response_format=None):
    while True:
        message_evaluator = llm_chat.chat(current_cov, None, response_format)

        if isinstance(message_evaluator, str):
            current_cov.append({"role": "assistant", "content": message_evaluator})
        else:
            current_cov.append({"role": "assistant", "content": message_evaluator.response})

        if "END" in current_cov[-1]["content"]:
            break
        # image_file = case["image"]
        image_file = Image.open(case["image"]).convert("RGB")
        output = eval_model(model_name, tokenizer, model, image_processor, context_len, type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "model_name": model_name,
            "query": message_evaluator if isinstance(message_evaluator, str) else message_evaluator.response,
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

        # output = 'There is no bird in the image'
        current_cov.append({"role": "user", "content": output})
        if isinstance(message_evaluator, str):
            print(f"examiner: {message_evaluator}")
        else:
            print(f"examiner: {message_evaluator.response}")
        print(f"vlm model: {output}")

        if isinstance(message_evaluator, str):
            to_save.append(
                {"object_index": object_round, "object_info": formatted_info, "prompt": message_evaluator,
                 "response": output}
            )
        else:
            to_save.append(
                {"object_index": object_round, "object_info": formatted_info, "prompt": message_evaluator.response,
                 "question_type": message_evaluator.question_type, "response": output}
            )
    return to_save, current_cov


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--p_mode", type=str, default="certainty")
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--outfile', type=str)
    args = parser.parse_args()

    samples = load_vg(args.debug)
    # with open('output/vg_samples.json', 'r') as json_file:
    #     samples = json.load(json_file)

    model_name, tokenizer, model, image_processor, context_len = load_model(args.model_path, args.model_base)
    model_path = args.model_path

    llm_chat = LLMChat(model_name="gpt-4o")

    object_round = 0
    print("starting conversation with model...")
    for sample in tqdm.tqdm(samples):
        to_save = []
        object_round += 1

        formatted_info = generate_fake_node(sample, llm_chat)
        print(f'Fake node:\n{formatted_info}')

        current_cov = [
            {"role": "system", "content": NODE_CONV_SYSTEM_PROMPT},
            *NODE_CONV_ICLs,
            {"role": "user", "content": formatted_info}
        ]

        to_save, current_cov = object_conv(sample, llm_chat, current_cov, object_round, formatted_info, to_save, model_name, tokenizer, model, image_processor, context_len, model_path, ResponseFormat)

        sample["conversations"] = to_save
        # sample["all_conversations"] = current_cov

        del sample["image"]
        with open(args.outfile, "w") as f:
            json.dump(samples, f, indent=4)
