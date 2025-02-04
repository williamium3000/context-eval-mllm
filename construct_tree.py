from nltk.corpus import wordnet as wn
import json
import re


def print_tree(node, level=-1):
    print_str = ' '.join(node['name'].split('.')[0].split('_'))

    indent = "  " * level + '-'
    print(f"{indent}{' '.join(node['name'].split('.')[0].split('_'))}")

    # for child in node.get("children", []):
    #     print_str += ', ' + print_tree(child, level + 1)
    #
    # if level == 0:
    #     print(level)
    #     if len(print_str.split(','))>1:
    #         print('1')

    return print_str


def get_object_map(object_file_path='object_synsets.json'):
    with open(object_file_path) as object_dict_file:
        object_dict = json.load(object_dict_file)
        object_dict = {key: value for key, value in object_dict.items()}
    return object_dict


def construct_relationship_tree(object_dict):
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


def is_vg_object(tree, response_word, gt_word):
    if response_word == gt_word:
        return True
    else:
        return is_father_child_relationship(tree, response_word, gt_word) or is_father_child_relationship(tree, gt_word, response_word)


def sentence_to_words(object_dict, sentence):
    object_pattern = r'\b(' + '|'.join(map(re.escape, object_dict.keys())) + r')\b'
    object_matches = re.findall(object_pattern, sentence)
    match_result = {match: object_dict[match] for match in object_matches}
    return match_result


if __name__ == '__main__':
    object_dict = get_object_map(object_file_path='object_synsets.json')
    relationship_tree = construct_relationship_tree(object_dict)

    # with open('class_tree_synset.json', 'w') as tree_file:
    #     json.dump(tree, tree_file, indent=2)
    # word_count = print_tree(tree)

    test_sentence = 'I have a candy.'
    gt_sentence = 'I have confections.'

    test_synset_match_result = sentence_to_words(object_dict, test_sentence)
    gt_synset_match_result = sentence_to_words(object_dict, gt_sentence)

    for gt_object, gt_object_class in gt_synset_match_result.items():
        for test_object, test_object_class in test_synset_match_result.items():
            print(test_object_class, gt_object_class)
            print(is_vg_object(relationship_tree, test_object_class, gt_object_class))
