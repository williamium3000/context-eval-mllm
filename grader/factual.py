import argparse
import json
import os
import tqdm
import torch

from factual_scene_graph.evaluation.evaluator import Evaluator
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser
import nltk
nltk.download('wordnet')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conv', type=str)
    parser.add_argument('--sg_model', type=str, default='lizhuang144/flan-t5-large-VG-factual-sg')
    parser.add_argument('--text_encoder', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--metric', type=str, default='all', choices=['all', 'set_match', 'spice', 'soft_spice'])
    parser.add_argument('--outdir', type=str, default='output/vg/sg')

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    output_path = os.path.join(args.outdir, os.path.basename(args.conv))


    # Load parser and evaluator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = SceneGraphParser(args.sg_model, device=device)
    evaluator = Evaluator(parser=parser, text_encoder_checkpoint=args.text_encoder, device=device, lemmatize=True)

    # Load conversation
    conversation_list = []
    ref_sg_list = []
    samples = json.load(open(args.conv, "r"))
    for sample in tqdm.tqdm(samples):
        conversation = sample["conversations"]
        sentence_list = []
        for turn in conversation:
            sentence_list += [sent for sent in turn["response"].strip().split('.') if len(sent)>0]

        # sentence_list = [turn["response"] for turn in conversation]

        sent_num = len(sentence_list)
        for attr in sample["attributes"]:
            if attr["attributes"] is None or len(attr["attributes"]) == 0:
                sentence_list += [attr['names'][0]]
            else:
                sentence_list += [f"{attribute} {attr['names'][0]}" for attribute in attr['attributes']]

        text_graph = parser.parse(sentence_list, beam_size=1, return_text=True, max_input_len=1000)
        conv_text_graph = ' , '.join(text_graph[:sent_num])
        conversation_list.append(conv_text_graph)

        ref_sg = [f"( {rel['subject']['names'][0]} , {rel['predicate'].lower()} , {rel['object']['names'][0]} )" for rel in sample["relationships"]]

        ref_sg += text_graph[sent_num:]
        reformatted_sg = ' , '.join(ref_sg)
        ref_sg_list.append([reformatted_sg])

    # Evaluate
    spice_scores, cand_graphs, ref_graphs = evaluator.evaluate(
        conversation_list,
        ref_sg_list,
        method='spice',
        beam_size=1,
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

    for i in range(len(samples)):
        text_sg = cand_graphs[i]
        samples[i]["lsg"] = text_sg

    with open(output_path, "w") as f:
        json.dump(samples, f, indent=4)
