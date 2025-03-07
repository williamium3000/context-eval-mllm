import argparse
import json
import os
import tqdm
import torch

from factual_scene_graph.evaluation.evaluator import Evaluator
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conv', type=str)
    parser.add_argument('--sg_model', type=str, default='lizhuang144/flan-t5-base-VG-factual-sg')
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
        conversation_to_be_evaluated = [turn["response"] for turn in conversation]

        conversation_list.append('\n'.join(conversation_to_be_evaluated))

        ref_sg = [f"( {rel['subject']['names'][0]} , {rel['predicate'].lower()} , {rel['object']['names'][0]} )" for rel in sample["relationships"]]
        region_phrases = [reg["phrase"] for reg in sample["regions"]]
        region_phrases_sg = parser.parse(region_phrases, beam_size=5, max_input_len=10000,
            max_output_len=256, return_text=True)
        ref_sg.extend(region_phrases_sg)
        
        for attr in sample["attributes"]:
            if attr["attributes"] is None or len(attr["attributes"]) == 0:
                continue
            ref_sg += [f"( {attr['names'][0]} , is , {attribute} )" for attribute in attr['attributes']]
        
        ref_sg = list(set(ref_sg))
        reformatted_sg = ' , '.join(ref_sg)
        ref_sg_list.append([reformatted_sg])
        
    
    # Evaluate
    spice_scores, cand_graphs, ref_graphs = evaluator.evaluate(
        conversation_list,
        ref_sg_list,
        method='spice',
        beam_size=5,
        batch_size=16,
        max_input_len=10000,
        max_output_len=256,
        return_graphs=True
    )
    print('SPICE scores:', sum(spice_scores) / len(spice_scores))

    if args.metric == 'all':
        set_match_scores = evaluator.evaluate(cand_graphs, ref_graphs, method='set_match', beam_size=1)
        print('Set Match scores:', sum(set_match_scores) / len(set_match_scores))

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
