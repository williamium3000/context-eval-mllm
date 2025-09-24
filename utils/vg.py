from datasets import load_dataset
import tqdm
import fsspec
import aiohttp

# Load the dataset with the specified download configuration
objects = load_dataset("visual_genome","objects_v1.2.0", trust_remote_code=True, split="train", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=7200)}})
attributes = load_dataset("visual_genome","attributes_v1.2.0", split="train", trust_remote_code=True)
relationships = load_dataset("visual_genome","relationships_v1.2.0", split="train", trust_remote_code=True)
regions = load_dataset("visual_genome","region_descriptions_v1.2.0", split="train", trust_remote_code=True)

# ['region_descriptions_v1.0.0', 
# 'region_descriptions_v1.2.0', 
# 'question_answers_v1.0.0', 
# 'question_answers_v1.2.0', 
# 'objects_v1.0.0', 'objects_v1.2.0', 'attributes_v1.0.0', 'attributes_v1.2.0', 
# 'relationships_v1.0.0', 'relationships_v1.2.0']

assert len(objects) == len(attributes) == len(relationships) == len(regions)

def load_sample_vg(idx):
    sample = objects[idx] 
    attrs = attributes[idx]["attributes"]
    rels = relationships[idx]["relationships"]
    regs = regions[idx]["regions"]
    assert len(sample["objects"]) == len(attrs)
    cur_objects = {}
    for i, obj in enumerate(attrs):
        object_id = obj["object_id"]
        obj["object_id"] = i
        cur_objects[object_id] = obj
    
    new_rels = []
    for rel in rels:
        del rel["relationship_id"]
        sub = rel["subject"]
        obj = rel["object"]
        
        new_sub_id = cur_objects[sub["object_id"]]
        new_obj_id = cur_objects[obj["object_id"]]
        rel["subject"] = new_sub_id
        rel["object"] = new_obj_id
        new_rels.append(rel)        

    scene_graph = {
        "objects": cur_objects,
        "relationships": new_rels,
        "regions": regs
    }
    del sample["objects"]
    sample["sg"] = scene_graph
    return sample


def load_vg(num_samples=None):
    all_img_ids = range(len(objects))
    if num_samples is not None:
        all_img_ids = all_img_ids[:num_samples]

    samples = []
    print(f"loading vg: total {len(all_img_ids)}")
    for img_id in tqdm.tqdm(all_img_ids):
        case = load_sample_vg(img_id)
        samples.append(case)
    return samples

def format_case_vg(case):
    formatted = "Instances:\n"
    H = case["height"]
    W = case["width"]
    
    sg = case["sg"]

    for ori_id, ins in sg["objects"].items():
        object_id = ins["object_id"]
        x, y, w, h = ins['x'], ins['y'], ins['w'], ins['h']
        x1, y1, x2, y2 = x / W, y / H, (x + w) / W, (y + h) / H
        if ins.get("attributes", []) is None or len(ins.get("attributes", [])) == 0:
            cur_attr = "none"
        else:
            attrs = ins.get("attributes", [])
            cur_attr = ", ".join(attrs)
        formatted += f"instance {object_id}, {ins['names'][0]}, bbox: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}), attributes: {cur_attr}\n"
    
    formatted += "\nRelation between the above instances:\n"
    for rel in sg["relationships"]:
        formatted += f"{rel['subject']['names'][0]} (instance {rel['subject']['object_id']}) {rel['predicate'].lower()} {rel['object']['names'][0]} (instance {rel['object']['object_id']})\n"
        
    return formatted
if __name__ == "__main__":
    print(format_case_vg(load_vg(num_samples=5)[0]))
