from vllm import LLM, SamplingParams

prompt = \
    """You will be given two words. Your task is to determin whether they refer to a similar thing. Here is some criteria to be consider:
1. whether they are synonym.
2. Whether one word is a subset of the other
3. You should generalize. If they generally refer to a similar thing, then respond 'yes'.
If any one of the above are met, then we consider the two words are similar and respond 'yes', otherwise respond 'no'. You should respond with 'yes' or 'no' ONLY without any additional texts.
Given words: '{word1}' and '{word2}'
Do they refer to a similar thing?""",

all_syns = []
data = open("test.txt").readlines()
for line in data:
    line = line.strip()
    cur_syn = [_.strip() for _ in line.split(",")]
    all_syns.append(cur_syn)
    

sampling_params = SamplingParams(temperature=1.2)

llm = LLM(model="/share/pretrain/llm/Qwen2.5-14B-Instruct")

all_prompts = []
for line in all_syns:
    root = line[0]
    for other in line[1:]:
        prompt_cur = prompt.format(word1=root, word2=other)
        all_prompts.append(prompt_cur)
bs = 32
outputs = []
for i in range(0, len(all_prompts) + bs - 1, bs):
    outputs += llm.generate(all_prompts[i:i+bs], sampling_params)

generated_text = [output.outputs[0].text for output in outputs]
correct = 0
cnt = 0
for t in generated_text:
    if "yes" in t:
       correct += 1
    cnt += 1

print(f"Accuracy: {correct/cnt}") 


# outputs = llm.generate(prompts, sampling_params)

# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(generated_text)