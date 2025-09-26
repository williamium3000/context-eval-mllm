import argparse
import json
import os
import re
import networkx as nx
import nltk
# nltk.download('wordnet')
import tqdm
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from graders.sg.llm_parser import parse_scene_graph
from utils.llm import LLMChat

lemmatizer = WordNetLemmatizer()

def word_to_synset(word, pos_tag=wordnet.NOUN):
    word = ' '.join(word.strip().lower().split())
    lemma_synset = set()

    # If the word consists of multiple parts, join them with an underscore
    word_split = word.split()
    if len(word_split) >= 2:
        word = "_".join(word_split)

    # # Add all synsets of the word to the set
    synonyms = wordnet.synsets(word, pos=pos_tag)
    for sys in synonyms:
        for lemma in sys.lemmas():
            lemma_synset.add(lemma.synset())

    return set().union(*[lemma_synset])


def similar_to_any(candidate, reference):
    candidate_synsets = word_to_synset(candidate)
    ref_synsets = word_to_synset(reference)
    return 0 if candidate_synsets & ref_synsets else 1


def node_subst_cost(n1_attrs, n2_attrs):
    name1 = n1_attrs.get("name")
    name2 = n2_attrs.get("name")

    if "attributes" in n1_attrs:
        attributes1 = n1_attrs["attributes"]
    else:
        attributes1 = []

    if "attributes" in n2_attrs:
        attributes2 = n2_attrs["attributes"]
    else:
        attributes2 = []
    cost = len(set(attributes1) ^ set(attributes2)) + similar_to_any(name1, name2)
    return cost


def edge_subst_cost(e1_attrs, e2_attrs):
    predicates1 = set(e1_attrs.get("predicates", []))
    predicates2 = set(e2_attrs.get("predicates", []))
    cost = 0 if predicates1 & predicates2 else 1
    return cost


def compare_scene_graphs(gt_graph, pred_graph):
    return nx.graph_edit_distance(
        gt_graph,
        pred_graph,
        node_subst_cost=node_subst_cost,
        edge_subst_cost=edge_subst_cost,
        node_del_cost=lambda attrs: 1 + len(attrs.get("attributes", [])),
        node_ins_cost=lambda attrs: 1 + len(attrs.get("attributes", [])),
        edge_del_cost=lambda attrs: 1,
        edge_ins_cost=lambda attrs: 1,
        timeout = 300
    )


def add_node_with_attributes(graph, name, attributes):
    if name not in graph.nodes:
        if attributes:
            # if not isinstance(attributes, list):
            #     print('attributes type', attributes)
            graph.add_node(name, name=name, attributes=attributes)
        else:
            graph.add_node(name, name=name)
    elif attributes and len(attributes):
        if "attributes" not in graph.nodes[name]:
            graph.nodes[name]["attributes"] = []

        for att in attributes:
            if att not in graph.nodes[name]["attributes"]:
                graph.nodes[name]["attributes"].append(att)


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default


def lemma_word(ori_word):
    tokens = ori_word.lower().split(' ')
    pos_tags = nltk.pos_tag(tokens)

    lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)).lower() for word, tag in pos_tags]
    lemma_word = ' '.join(lemmas)
    return lemma_word


def scene_graph_to_nx(relationships, attributes, objects):
    graph = nx.DiGraph()

    # Index attributes by object_id
    attr_map = {attr["object_id"]: {k: v for k, v in attr.items() if k != "object_id"}
                for attr in attributes}

    # Add nodes edges from relationships
    for rel in relationships:
        subj = rel["subject"].copy()
        obj = rel["object"].copy()

        sub_name = lemma_word(subj["names"][0])
        obj_name = lemma_word(obj["names"][0])
        # Merge attributes if available
        if subj["object_id"] in attr_map.keys():
            subj.update(attr_map[subj["object_id"]])
        if obj["object_id"] in attr_map:
            obj.update(attr_map[obj["object_id"]])

        # Add nodes
        add_node_with_attributes(graph, sub_name, subj.get("attributes", []))
        add_node_with_attributes(graph, obj_name, obj.get("attributes", []))

        # Add edge
        pred = lemma_word(rel['predicate'])
        if graph.has_edge(sub_name, obj_name):
            if pred not in graph[sub_name][obj_name]["predicates"]:
                graph[sub_name][obj_name]["predicates"].append(pred)
        else:
            graph.add_edge(sub_name, obj_name, predicates=[pred])

    # Add attribute-only nodes
    for obj_id, attrs in attr_map.items():
        att_name = lemma_word(attrs["names"][0])
        if att_name not in graph.nodes and attrs["attributes"]:
            graph.add_node(att_name, name=att_name, attributes=attrs["attributes"])
        elif att_name not in graph.nodes:
            graph.add_node(att_name, name=att_name)
        elif attrs["attributes"] and len(attrs["attributes"]):
            if "attributes" not in graph.nodes[att_name]:
                graph.nodes[att_name]["attributes"] = []

            for att in attrs["attributes"]:
                if att not in graph.nodes[att_name]["attributes"]:
                    graph.nodes[att_name]["attributes"].append(att)
        # else:
        #     print(attrs)

    # Add object-only nodes
    for obj in objects:
        obj_name = lemma_word(obj["names"][0])
        if obj_name not in graph.nodes:
            graph.add_node(obj_name, name=obj_name)

    return graph


def parse_scene_graph_string(s):
    triplets = []
    for match in re.findall(r"\((.*?)\)", s):
        parts = [p.strip() for p in match.split(",")]
        if len(parts) == 3:
            triplets.append(tuple(parts))
        elif len(parts) == 1:
            triplets.append((parts[0],))
        else:
            raise ValueError(f"Unexpected format: ({match})")
    return triplets


def build_graph_from_string(s):
    graph = nx.DiGraph()
    triplets = parse_scene_graph_string(s)

    for triple in triplets:
        if len(triple) == 3:
            subj, pred, obj = triple
            subj, pred, obj = lemma_word(subj), lemma_word(pred), lemma_word(obj)
            if subj not in graph.nodes:
                graph.add_node(subj, name=subj)

            if pred == "be":
                if "attributes" not in graph.nodes[subj]:
                    graph.nodes[subj]["attributes"] = []
                if obj not in graph.nodes[subj]["attributes"]:
                    graph.nodes[subj]["attributes"].append(obj)
            else:
                if obj not in graph.nodes:
                    graph.add_node(obj, name=obj)

                if graph.has_edge(subj, obj):
                    if pred not in graph[subj][obj]["predicates"]:
                        graph[subj][obj]["predicates"].append(pred.lower())
                else:
                    graph.add_edge(subj, obj, predicates=[pred.lower()])
        else:
            node = triple[0]
            node = lemma_word(node)
            if node not in graph.nodes:
                graph.add_node(node, name=node)

    return graph


def show_graph(graph, figsize=(18, 14), layout="spring"):
    plt.figure(figsize=figsize)

    # choose layout
    if layout == "spring":
        pos = nx.spring_layout(graph, k=1.5, iterations=100)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    elif layout == "shell":
        pos = nx.shell_layout(graph)
    else:
        pos = nx.spring_layout(graph, k=1.5)

    # draw nodes
    nx.draw_networkx_nodes(
        graph, pos,
        node_size=2000,
        node_color="skyblue",
        edgecolors="black"
    )

    # draw edges with curve
    nx.draw_networkx_edges(
        graph, pos,
        arrows=True,
        arrowstyle="->",
        arrowsize=50,
        width=2,
        edge_color="gray",
        connectionstyle="arc3,rad=0.2",
    min_target_margin = 20,  # push arrowhead out from target node
    min_source_margin = 15
    )

    # node labels
    labels = {n: data.get("name", n) for n, data in graph.nodes(data=True)}
    nx.draw_networkx_labels(
        graph, pos,
        labels,
        font_size=14,
        font_weight="bold"
    )

    # edge labels
    edge_labels = {
        (u, v): ",".join(data.get("predicates", []))
        for u, v, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        graph, pos,
        edge_labels=edge_labels,
        font_size=12,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conv_script', type=str, default="output/vg/sg/caption_chatgpt_lsg.json")
    parser.add_argument('--outdir', type=str, default="output/vg/sg")
    parser.add_argument('--output_file', type=str, default="caption_chatgpt_lsg_dis.json")
    args = parser.parse_args()

    agent = LLMChat("gpt-4o-mini")
    samples = json.load(open(args.conv_script, "r"))

    os.makedirs(args.outdir, exist_ok=True)
    output_path = os.path.join(args.outdir, args.output_file)

    dist_list = []
    idx = 0
    for sample in tqdm.tqdm(samples):
        idx += 1
        sg_dict, unique_sg = parse_scene_graph(sample, agent)
        sample["lsg"] = list(unique_sg)
        pred_sg = ' , '.join(list(unique_sg))

        relationships = sample["relationships"]
        objects = sample["objects"]
        attributes = sample["attributes"]

        gt_graph = scene_graph_to_nx(relationships, attributes, objects)
        pred_graph = build_graph_from_string(pred_sg)
        # pred_graph = build_graph_from_string(' , '.join(sample["unique_sg"]))

        # show_graph(gt_graph, layout="shell")
        # show_graph(pred_graph, layout="shell")

        dist_score = compare_scene_graphs(gt_graph, pred_graph)
        sample["dist_score"] = dist_score
        dist_list.append(dist_score)
        print(f'Sample {idx} distance', dist_score)

    with open(output_path, "w") as file:
        json.dump(samples, file, indent=4)

    print('Min distance', sum(dist_list)/len(dist_list))
