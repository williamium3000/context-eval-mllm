from PIL import Image
from infer.infer_llava import eval_model


PARAPHRASE_ICLs = [
    {"role": "user", "content": "What catches your eye about the metal bench placed in the park?"},
    {"role": "assistant", "content": "What stands out to you about the metal bench located in the park?"},
    {"role": "user", "content": "Can you describe the big tree growing close to the bench?"},
    {"role": "assistant", "content": "How would you describe the large tree near the bench?"},
    {"role": "user", "content": "How would you describe the surface and surroundings around the bench?"},
    {"role": "assistant", "content": "What does the ground and area around the bench look like to you?"},
]


PARAPHRASE_SYSTEM_PROMPT = \
'''Task: Given a list of questions, your goal is to paraphrase each question while preserving its original meaning. 
Avoid changing the core intent or level of detail, but feel free to vary the wording and sentence structure for naturalness or clarity.

Requirements
1. For each question, generate one paraphrased version by rephrasing yes/no questions into wh-questions, using synonyms, altering the sentence structure or changing focus order (e.g., object before relation or relation before object).
2. Ensure the paraphrased version is grammatically correct and sounds natural.
3. Do not add, remove, or alter the core information or intent.
4. Do not reference the original question or indicate that it was paraphrased.
5. Output the paraphrased question only.
'''


def paraphrase_conv(case, llm_chat, object_index, node_info, qa_list, model_name, tokenizer, model, image_processor, context_len, model_path):
    to_save = []

    for index, content in enumerate(qa_list):
        if index % 2 == 0:
            conversations = [
                {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
                *PARAPHRASE_ICLs,
                {"role": "user", "content": content["content"]}]

            message_evaluator = llm_chat.chat(conversations, None)

            # image_file = case["image"]
            image_file = Image.open(case["image"]).convert("RGB")
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

            print(f"examiner: {message_evaluator}")
            print(f"vlm model: {output}")

            to_save.append(
                {"repeated_object_index": object_index, "object_info": node_info, "prompt": message_evaluator,
                 "response": output}
            )

    return to_save

