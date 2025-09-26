import os
import argparse
import json
from collections import defaultdict
import numpy as np
import tqdm
import torch
# from utils.construct_tree import get_object_map
# from graders.factual import is_physical_object
from utils.llm import LLMChat
from factual_scene_graph.evaluation.evaluator import Evaluator
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn


def get_object_map(object_file_path='object_synsets.json'):
    with open(object_file_path) as object_dict_file:
        object_dict = json.load(object_dict_file)
        object_dict = {key.split('.')[0]: value for key, value in object_dict.items()}

    filtered_object_dict = {key: value for key, value in object_dict.items() if is_physical_object(value)}
    return filtered_object_dict


device = "cuda" if torch.cuda.is_available() else "cpu"
object_dict = get_object_map(object_file_path='data/filtered_object_synsets_final.json')


filtered_non_phy_words = ["city", "image", "city life", "landscape", "light", "time", "diversity", 'traffic rule', 'cityscape','activity','scene','environment', 'workspace', 'area']
filtered_attribute = ["visible", 'well-organized']


def get_hypernym_tree(synset):
    hypernyms = set()

    def traverse(syn):
        if syn not in hypernyms:
            hypernyms.add(syn)
            for hypernym in syn.hypernyms():
                traverse(hypernym)

    traverse(synset)
    return hypernyms


def is_physical_object(subject_word, sg_tuple, object_dict):
    if subject_word.lower() in filtered_non_phy_words:
        return False
    elif subject_word.lower() in object_dict.keys() or subject_word in object_dict.values():
        return True
    else:
        phrase = sg_tuple.replace(',', '').lower()
        best_synset = lesk(phrase, subject_word, pos=wn.NOUN)
        if best_synset:
            syn = wn.synset(best_synset.name())
            hypernym_tree = get_hypernym_tree(syn)
            if any(hypernym.name().startswith(
                    ("physical_entity.n.", "people.n.", "person.n.", "vegetation.n", "building.n", "tree.n")) for
                   hypernym in hypernym_tree):
                return True

        for word in subject_word.split(' '):
            best_synset = lesk(phrase, word, pos=wn.NOUN)

            if best_synset:
                syn = wn.synset(best_synset.name())
                hypernym_tree = get_hypernym_tree(syn)
                if any(hypernym.name().startswith(("physical_entity.n.", "people.n.", "person.n.", "vegetation.n", "building.n", "tree.n")) for hypernym in hypernym_tree):
                    # print(True, syn)
                    return True
        # print(False, syn)
        return False


def refine_output(response):
    before_match_parsing = []
    if 'none' in response.lower():
        return []

    response = response.replace('>,', '>')
    for a in response.split("<")[1:]:
        a = a.split(",")
        # if len(a) < 3:
        #     continue
        if len(a) == 3 and a[1].strip() not in ['is', 'was', 'are', 'were']:
            sub, pred, obj = a[0].strip(), a[1].strip(), a[2].split(">")[0].strip()
            sub_status = is_physical_object(sub, ' '.join(a), object_dict)
            obj_status = is_physical_object(obj, ' '.join(a), object_dict)
            if sub_status and obj_status:
                before_match_parsing.append([sub, pred, obj])
            elif obj_status:
                before_match_parsing.append([obj])
            elif sub_status:
                before_match_parsing.append([sub])

        elif len(a) == 1:
            sub = a[0].split(">")[0].strip()
            if is_physical_object(sub, ' '.join(a), object_dict):
                before_match_parsing.append([sub])
        elif len(a) == 3 and a[1].strip() in ['is', 'was', 'are', 'were']:
            sub, pred, adj = a[0].strip(), a[1].strip(), a[2].split(">")[0].strip()
            if is_physical_object(sub, ' '.join(a), object_dict):
                before_match_parsing.append([sub, pred, adj])
        else:
            print('Error format', a)

    return before_match_parsing


def parse_scene_graph(sample, agent):
    conversation = sample["conversations"]

    output_dict = {}
    unique_sg = []
    for index, turn in enumerate(conversation):
        vlm_response = turn["response"]

        response_list = [question for question in vlm_response.strip('\n').split('\n') if len(question)]

        lsg_list = []
        for question in response_list:
            LSG_PROMPT = f"""From the given sentence, the task is to extract scene graphs formed as <subject, predicate, object>, <object, is, attribute> or <object>. Note that the subject is the physical entity or noun that performs the action or is being described, and the object is the physical entity or noun that is affected by the action or is receiving the action. The predicate is a verb or adjective without auxiliary verb, and is represented without the tense (e.g., are, being). The attribute is a physical quality or characteristic (typically an adjective) directly modifying an object or entity (e.g., <jacket, is, red>, <wall, is, wooden>).
Instructions:
- If an object has no attributes or relations, output it directly in the form <object>.
- Do **not** extract scene graphs involving:
  - Objects, subjects or relations that are negated (e.g., "There is no man...")
  - Non-physical entities (e.g., "atmosphere", "conversation") in subject or object
  - Entities or relations that are **speculative or inferred** from other clues rather than explicitly described as visible (e.g., "could indicate", "might suggest", "possibly", "likely")
  - Abstract scene descriptions that cannot be directly grounded in physical objects or traits (e.g., "scene is urban", "shirt adds pop of color", "building contributes to atmosphere")
  - Attributes that are subjective or stylistic rather than physical (e.g., "beautiful", "cozy", "futuristic" when not tied to tangible features)
  - Statements about effects, purposes, or benefits rather than direct physical description (e.g., "contributes to convenience", "supports community well-being")

### Examples
Sentence: "A slice of bread is covered with a sour cream and guacamole."
Triplets: <bread, covered with, sour cream>, <bread, covered with, guacamole>

Sentence: "A beautiful woman walking a dog on top of a beach."
Triplets: <woman, walking with, dog>, <woman, on, beach>, <dog, on, beach>

Sentence: "Four clocks sitting on a floor next to a woman's feet."
Triplets: <clock, sitting on, floor>, <clock, next to, feet>

Sentence: "One person sits in a chair looking at her phone while another rests on the couch."
Triplets: <person, sits in, chair>, <person, looking at, phone>, <person, rests on, couch>

Sentence: "A lady and a child near a park bench with kites and ducks flying in the sky and on the ground."
Triplets: <lady, near, park bench>, <child, near, park bench>, <kites, flying in, sky>, <ducks, on, ground>

Sentence: "Two men sit on a bench near the sidewalk and one of them talks on a cell phone."
Triplets: <men, sit on, bench>, <bench, near, sidewalk>, <man, talks on, phone>

Sentence: "There is no man wearing a red jacket in the image."  
Triplets: (none)

Sentence: "A man wearing a red jacket is in the image."  
Triplets: <man, wearing, jacket>, <jacket, is, red>

Sentence: "The carpet on the wooden floor is blue."  
Triplets: <carpet, on, floor>, <floor, is, wooden>, <carpet, is, blue>

Sentence: "There are several cars parked along the street, and a bicycle is also visible."  
Triplets: <cars, parked on, street>, <bicycle>

Sentence: "People on the street are engaged in various activities and interactions."  
Triplets: <people, on, street>

Sentence: "The traffic light indicates that the street is regulated for vehicle and pedestrian safety."  
Triplets: <traffic light>, <street>

### Now extract triplets from the following sentence:
Sentence: \"{question}\"\nTriplets:
            """

            messages = [
                        {"role": "system", "content": "From the given sentence, your task is to extract meaningful triplets formed as <subject, predicate, object>."},
                        {"role": "user", "content": LSG_PROMPT.strip()}
            ]

            parsing_result = agent.chat(messages, None, temperature=0)
            # print('SG', parsing_result)
            refined_sg = refine_output(parsing_result)

            lsg_list += refined_sg
            unique_sg += [f"( {' , '.join(sg)} )" for sg in refined_sg]
            # print('Question:', question)
            # print('Unique SG', [f"( {' , '.join(sg)} )" for sg in refined_sg])
        output_dict[index] = lsg_list
        
    return output_dict, set(unique_sg)


def get_annot_sg(sample):
    ref_sg = []
    # Add relationship info
    for rel in sample["relationships"]:
        rel_info = f"( {rel['subject']['names'][0].lower()} , {rel['predicate'].lower()} , {rel['object']['names'][0].lower()} )"
        ref_sg += [rel_info]

    # Add attribute info
    for attr in sample["attributes"]:
        if attr["attributes"] is None or len(attr["attributes"]) == 0:
            continue
        else:
            for attribute in attr['attributes']:
                att_info = f"( {attr['names'][0]}, is , {attribute} )"
                ref_sg += [att_info]

    # Add object num
    object_num_dict = {}
    for object_name in sample["objects"]:
        # if object_name["names"][0] not in object_num_dict.keys():
        #     object_num_dict[object_name["names"][0]] = 1
        # else:
        #     object_num_dict[object_name["names"][0]] = +1

        ref_sg += [f'( {object_name["names"][0]} )']

    return ref_sg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conv_script', type=str, default="output/vg/caption.json")
    parser.add_argument('--metric', type=str, default='all', choices=['all', 'set_match', 'spice', 'soft_spice'])
    parser.add_argument('--text_encoder', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sample_num', type=int, default=20)
    parser.add_argument('--outdir', type=str, default="output/vg/sg")
    args = parser.parse_args()

    agent = LLMChat("gpt-4o-mini")

    os.makedirs(args.outdir, exist_ok=True)
    output_path = os.path.join(args.outdir, "caption_chatgpt_lsg_new.json")

    samples = json.load(open(args.conv_script, "r"))

    ref_sg_list = []
    anno_conv_list = []
    print("Start parsing...")
    for sample in tqdm.tqdm(samples[:args.sample_num]):
        sg_dict, unique_sg = parse_scene_graph(sample, agent)
        sample["sg"] = sg_dict
        sample["unique_sg"] = list(unique_sg)
        anno_conv_list += [' , '.join(list(unique_sg))]

        # Add annotation info
        ref_sg = get_annot_sg(sample)
        ref_sg_list.append([' , '.join(set(ref_sg))])

    with open(output_path, "w") as file:
        json.dump(samples[:args.sample_num], file, indent=4)

    evaluator = Evaluator(parser=parser, text_encoder_checkpoint=args.text_encoder, device=device, lemmatize=False)
    spice_scores, cand_graphs, ref_graphs = evaluator.evaluate(
        anno_conv_list,
        ref_sg_list,
        method='spice',
        beam_size=args.beam_size,
        batch_size=128,
        max_input_len=10000,
        max_output_len=256,
        return_graphs=True
    )
    print('SPICE scores:', sum(spice_scores) / len(spice_scores))

    if args.metric == 'all':
        # set_match_scores = evaluator.evaluate(cand_graphs, ref_graphs, method='set_match', beam_size=1)
        # print('Set Match scores:', sum(set_match_scores) / len(set_match_scores))

        soft_spice_scores = evaluator.evaluate(cand_graphs, ref_graphs, method='soft_spice', beam_size=1)
        print('Soft-SPICE scores:', sum(soft_spice_scores) / len(soft_spice_scores))
    else:
        soft_spice_scores = evaluator.evaluate(cand_graphs, ref_graphs, method=args.metric, beam_size=1)
        print(f'{args.metric} scores:', sum(soft_spice_scores) / len(soft_spice_scores))

    for i in range(len(samples[:args.sample_num])):
        text_sg = cand_graphs[i]
        # samples[i]["lsg"] = text_sg
        samples[i]["spice"] = spice_scores[i]
        samples[i]["soft_spice"] = soft_spice_scores[i]

    with open(output_path, "w") as f:
        json.dump(samples, f, indent=4)
