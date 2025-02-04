from nltk.stem import *
from nltk.corpus import wordnet as wn
import json
import argparse
import os
import json
import numpy as np
import nltk
import re


def sentence_to_words(object_dict, sentence):
    object_pattern = r'\b(' + '|'.join(map(re.escape, object_dict.keys())) + r')\b'
    object_matches = re.findall(object_pattern, sentence)
    match_result = [(match, object_dict[match]) for match in object_matches]
    return match_result

def compute_chair(caps, total_synsets, syn_tree):

    '''
    Given ground truth objects and generated captions, determine which sentences have hallucinated words.
    '''

    num_caps = 0.
    num_hallucinated_caps = 0.
    hallucinated_word_count = 0.
    coco_word_count = 0.
    
    output = {
        'sentences': [], "vg_words": [], 
        "vg_hallucinated_words": [],
        "object_in_gts": [],
        "gt_objects": [],
        "coverage": []} 

    for i, cap_eval in enumerate(caps):

        cap = cap_eval['caption']
        imid = cap_eval['image_id']

        gt_synsets = []
        # objects
        for obj in cap_eval['objects']:
            gt_synsets.extend(obj['synsets'])
        # attributes
        for attr in cap_eval['attributes']:
            gt_synsets.extend(attr['synsets'])
        # relationships
        for rel in cap_eval['relationships']:
            gt_synsets.extend(rel["subject"]['synsets'])
            gt_synsets.extend(rel["object"]['synsets'])
        gt_synsets = list(set(gt_synsets))
        output_objects = sentence_to_words(total_synsets, cap)
        
        raw_words = nltk.word_tokenize(cap.lower())
        # raw_words = [singularize(w) for w in raw_words]
        
        cap_dict = {'image_id': cap_eval['image_id'], 
                    'caption': cap,
                    'vg_hallucinated_words': [],
                    'vg_gt_words': gt_synsets,
                    'vg_generated_words': [_[0] for _ in output_objects],
                    "words": raw_words
                    }
        # print(cap_eval)
        cap_dict['metrics'] = {'CHAIRs': 0,
                                'CHAIRi': 0}

        #count hallucinated words
        coco_word_count += len(raw_words)
        hallucinated = False
        mscoco_words_i = []
        mscoco_hallucinated_words_i = []
        object_in_gts = []
        
        for word, syn_class in output_objects:
            
            mscoco_words_i.append((word, syn_class))
            if is_hallucinate(syn_tree, syn_class, gt_synsets):
                hallucinated_word_count += 1 
                cap_dict['vg_hallucinated_words'].append((word, syn_class))
                
                mscoco_hallucinated_words_i.append((word, syn_class))
                hallucinated = True    
            else:
                object_in_gts.append(syn_class)

        mscoco_words_i = list(set(mscoco_words_i))
        mscoco_hallucinated_words_i = list(set(mscoco_hallucinated_words_i))
        
        output['vg_words'].extend(mscoco_words_i)
        output['vg_hallucinated_words'].extend(mscoco_hallucinated_words_i)
        output['object_in_gts'].extend(list(set(object_in_gts)))
        output['gt_objects'].extend(gt_synsets)
        
        if len(gt_synsets) > 0:
            output["coverage"].append(len(list(set(object_in_gts))) / len(gt_synsets))
        
        #count hallucinated caps
        num_caps += 1
        if hallucinated:
            num_hallucinated_caps += 1

        cap_dict['metrics']['CHAIRs'] = int(hallucinated)
        cap_dict['metrics']['CHAIRi'] = 0.
        if len(raw_words) > 0:
            cap_dict['metrics']['CHAIRi'] = len(cap_dict['vg_hallucinated_words'])/float(len(raw_words))

        output['sentences'].append(cap_dict)

    chair_s = (num_hallucinated_caps/num_caps)
    chair_i = (hallucinated_word_count/coco_word_count)
    chair_i_v2 = len(output['vg_hallucinated_words'])/len(output['vg_words'])
    coverage_avg = np.mean(output["coverage"])
    coverage_all = len(output['object_in_gts']) / len(output['gt_objects'])
    output['overall_metrics'] = {
                                    'CHAIRs': chair_s,
                                    'CHAIRi': chair_i,
                                    "CHAIRi_v2": chair_i_v2,
                                    "Coverage_avg": coverage_avg,
                                    "Coverage_all": coverage_all
                                    }

    return output 


def save_hallucinated_words(cap_file, cap_dict): 
    tag = cap_file.split('/')[-1] 
    with open(os.path.join(os.path.dirname(cap_file), f'hallucinated_words_{tag}'), 'w') as f:
        json.dump(cap_dict, f)

def print_metrics(hallucination_cap_dict, quiet=False):
    sentence_metrics = hallucination_cap_dict['overall_metrics']
    metric_string = "%0.01f\t%0.01f\t%0.01f\t%0.01f\t%0.01f" %(
                                                    sentence_metrics['CHAIRs']*100,
                                                    sentence_metrics['CHAIRi']*100,
                                                    sentence_metrics['CHAIRi_v2']*100,
                                                    sentence_metrics['Coverage_avg']*100,
                                                    sentence_metrics['Coverage_all']*100
                                                  )

    if not quiet:
        print("CHAIRs\tCHAIRi\tCHAIRi_v2\tCoverage-avg\tCoverage-all")
        print(metric_string)

    else:
        return metric_string





def synset_tree(object_dict):
    synset_name_list = list(set(object_dict.values()))

    # with open('word_synset.txt', "r") as word_file:
    #     word_list = word_file.readlines()
    #     word_list = [word.strip() for word in word_list]

    tree = {"name": "root", "children": []}
    added_nodes = {}

    def find_or_create_node(name):
        if name in added_nodes.keys():
            return added_nodes[name]
        new_node = {"name": name, "children": []}
        added_nodes[name] = new_node
        return new_node

    # Build the tree
    for synset_name in synset_name_list:
        synset = wn.synset(synset_name)

        word = synset_name.split('.')[0]
        word_node = find_or_create_node(synset_name)

        skip = False
        for hypernym in synset.hypernyms():
            hypernym_name = hypernym.name()
            if hypernym_name in synset_name_list:
                # if len(synset.hypernyms()) > 1:
                #     print(word, synset.hypernyms())

                hypernym_node = find_or_create_node(hypernym_name)
                if hypernym_name not in [object_dict["name"] for object_dict in hypernym_node["children"]]:
                    hypernym_node["children"].append(word_node)
                    # print(f'Word: {word}, Hyper: {hypernym_name}')
                    skip = True

        if not skip:
            if synset_name not in [object_dict["name"] for object_dict in tree["children"]]:
                tree["children"].append(word_node)

    return tree


def is_father_child_relationship(tree, father_name, child_name):
    if not tree:
        return False

    if tree["name"] == father_name:
        for child in tree["children"]:
            if child["name"] == child_name:
                return True

    for child in tree["children"]:
        if is_father_child_relationship(child, father_name, child_name):
            return True

    return False


def is_hallucinate(tree, response_word, gt_words):
    
    for gt_word in gt_words:
        if (response_word == gt_word) or is_father_child_relationship(tree, response_word, gt_word) or is_father_child_relationship(tree, gt_word, response_word):
            return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cap_file", type=str)
    parser.add_argument("object_synsets", type=str)
    args = parser.parse_args()
    
    with open(args.object_synsets) as object_dict_file:
        object_dict = json.load(object_dict_file)
        object_dict = {key: value for key, value in object_dict.items()}
    
    syn_tree = synset_tree(object_dict)
    
    data = json.load(open(args.cap_file, 'r'))
    chair_result = compute_chair(data, object_dict, syn_tree)
    print_metrics(chair_result)