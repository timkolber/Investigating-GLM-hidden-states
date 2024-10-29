import json
import os
from copy import deepcopy

import pandas as pd
from transformers import AutoModel, AutoTokenizer

from GraphLanguageModels.models.graph_T5.wrapper_functions import r2nl

splits = ["train", "dev", "test"]

data = {split: {} for split in splits}  #

tokenizer = AutoTokenizer.from_pretrained("plenz/GLM-t5-large")
model = AutoModel.from_pretrained("plenz/GLM-t5-large", trust_remote_code=True)

for split in splits:
    radius_dict = {radius: {} for radius in range(1, 6)}
    data[split] = radius_dict
    for radius in range(1, 6):
        graphs = []
        labels = []
        with open(
            f"/home/students/kolber/seminars/kolber/data/radius={radius}/{split}_graphs.jsonl",
            "r",
        ) as gf:
            for line in gf:
                json_line = json.loads(line)
                graphs.append(json_line)

        with open(
            f"/home/students/kolber/seminars/kolber/data/radius={radius}/{split}_labels.jsonl",
            "r",
        ) as lf:
            for line in lf:
                labels.append(line.strip())

        data[split][radius]["graphs"] = graphs
        data[split][radius]["labels"] = labels

triplet_func_to_idx = {"subject": 0, "relation": 1, "object": 2}


def create_target_prompt(graph):
    target_prompt = "summarize: "
    for triplet in graph:
        if "<extra_id_0>" not in triplet:
            continue
        for element in triplet:
            target_prompt += f"{element} "
    # target_prompt = target_prompt.replace("<extra_id_0>", "?")
    return target_prompt.strip()


def find_source_position(graph):
    input_data_target = model.encoder.data_processor.encode_graph(
        tokenizer=tokenizer,
        g=graph,
        how="global",
    )

    inp_target = model.encoder.data_processor.to_batch(
        data_instances=[input_data_target],
        tokenizer=tokenizer,
        max_seq_len=None,
        device="cpu",
    )
    source_position = (
        inp_target["input_ids"]
        .tolist()[0]
        .index(tokenizer.convert_tokens_to_ids("<extra_id_0>"))
    )
    return source_position


def find_target_position(text):
    tokenized_text = tokenizer.tokenize(text, add_special_tokens=True)
    return tokenized_text.index("<extra_id_0>")


def create_graphs_and_labels(graphs, labels, mask_item="subject"):
    new_labels = []
    new_graphs = []
    triplet_mask_idx = triplet_func_to_idx[mask_item]
    for graph, label in zip(graphs, labels):
        new_graph = deepcopy(graph)
        for triplet_idx, triplet in enumerate(graph):
            for element_idx, element in enumerate(triplet):
                if (
                    "<mask>" not in new_graph[triplet_idx]
                    and "extra_id_0" not in new_graph[triplet_idx]
                ):
                    new_graph[triplet_idx][1] = r2nl(new_graph[triplet_idx][1])
                if element == "<mask>":
                    new_graph[triplet_idx][element_idx] = r2nl(label)
                    new_label = r2nl(new_graph[triplet_idx][triplet_mask_idx])
                    new_labels.append(new_label)
                    new_graph[triplet_idx][triplet_mask_idx] = "<extra_id_0>"
        new_graphs.append(new_graph)

    return new_graphs, new_labels


for mask_triplet_element in triplet_func_to_idx.keys():
    for split in splits:
        for radius in range(1, 6):
            graphs = deepcopy(data[split][radius]["graphs"])
            labels = deepcopy(data[split][radius]["labels"])
            new_graphs, new_labels = create_graphs_and_labels(
                graphs, labels, mask_triplet_element
            )
            target_prompts = [create_target_prompt(graph) for graph in new_graphs]
            target_positions = [
                find_target_position(target_prompt) for target_prompt in target_prompts
            ]
            source_positions = [find_source_position(graph) for graph in new_graphs]
            df = pd.DataFrame(
                {
                    "graph": new_graphs,
                    "label": new_labels,
                    "target_prompt": target_prompts,
                    "target_position": target_positions,
                    "source_position": source_positions,
                }
            )
            # make directory
            os.makedirs(
                f"/home/students/kolber/seminars/kolber/data/radius={radius}/{mask_triplet_element}_masked",
                exist_ok=True,
            )
            df.to_csv(
                f"/home/students/kolber/seminars/kolber/data/radius={radius}/{mask_triplet_element}_masked/{split}.csv",
                index=False,
                mode="w",  # to ensure that the file is overwritten
            )
