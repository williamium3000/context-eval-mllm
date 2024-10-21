from datasets import load_dataset
import tqdm
# Create a download configuration with resume_download set to True

# Load the dataset with the specified download configuration
objects = load_dataset("visual_genome","objects_v1.2.0", split="train")
attributes = load_dataset("visual_genome","attributes_v1.2.0", split="train")
relationships = load_dataset("visual_genome","relationships_v1.2.0", split="train")
regions = load_dataset("visual_genome","region_descriptions_v1.2.0", split="train")

# ['region_descriptions_v1.0.0', 
# 'region_descriptions_v1.2.0', 
# 'question_answers_v1.0.0', 
# 'question_answers_v1.2.0', 
# 'objects_v1.0.0', 'objects_v1.2.0', 'attributes_v1.0.0', 'attributes_v1.2.0', 
# 'relationships_v1.0.0', 'relationships_v1.2.0']

assert len(objects) == len(attributes) == len(relationships) == len(regions)

def load_sample_vg(idx):
    sample = objects[idx]
    sample["attributes"] = attributes[idx]["attributes"]
    sample["relationships"] = relationships[idx]["relationships"]
    sample["regions"] = regions[idx]["regions"]
    return sample


def load_vg(debug=False):
    all_img_ids = range(len(objects))
    if debug:
        all_img_ids = all_img_ids[:5]

    samples = []
    print(f"loading vg: total {len(all_img_ids)}")
    for img_id in tqdm.tqdm(all_img_ids):
        case = load_sample_vg(img_id)
        samples.append(case)
    return samples

def format_case_vg(case):
    formatted = "Instances:\n"
    h = case["height"]
    w = case["width"]
        
    objects = case["objects"]
    attributes = {}
    for attr in case["attributes"]:
        if attr["attributes"] is None or len(attr["attributes"]) == 0:
            continue
        if attr["object_id"] not in attributes:
            attributes[attr["object_id"]] = attr["attributes"]
        else:
            attributes[attr["object_id"]] += attr["attributes"]
    
    relationships = case["relationships"]
    for ins in objects:
        object_id = ins["object_id"]
        x, y, w, h = ins['x'], ins['y'], ins['w'], ins['h']
        x1, y1, x2, y2 = x / w, y / h, (x + w) / w, (y + h) / h
        cur_attr = ", ".join(attributes.get(object_id, []))
        formatted += f"{ins['names'][0]}, bbox: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}), attributes: {cur_attr}\n"
    formatted += "\nRelation between the above instances:\n"
    for rel in relationships:
        formatted += f"{rel['subject']['names'][0]} {rel['predicate'].lower()} {rel['object']['names'][0]}\n"
    return formatted
if __name__ == "__main__":
    print(format_case_vg(load_vg(debug=True)[0]))
