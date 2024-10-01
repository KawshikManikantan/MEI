import os
from os import path
from collections import defaultdict
import jsonlines


def get_major_entities(input_jsonlines):
    if path.isdir(input_jsonlines):
        doc_names = [
            file_name[:-6]
            for file_name in os.listdir(input_jsonlines)
            if file_name.endswith(".jsonl")
        ]
        major_entities = {}
        for document in doc_names:
            with jsonlines.open(
                os.path.join(input_jsonlines, document + ".jsonl")
            ) as reader:
                major_entities[document] = defaultdict(list)
                for obj in reader:
                    for key in obj:
                        major_entities[document][key].append(obj[key])
    else:
        ## This is for sota models
        major_entities = {}
        major_entities_sota = read_jsonl(input_jsonlines)
        for document in major_entities_sota:
            major_entities[document] = {}
            mention_inds = major_entities_sota[document]["mention_inds"]
            mention_strs = major_entities_sota[document]["mention_strs"]
            cluster_inds = major_entities_sota[document]["cluster_inds"]
            if mention_inds != []:
                mention_inds, mention_strs, cluster_inds = zip(
                    *sorted(zip(mention_inds, mention_strs, cluster_inds))
                )
            major_entities[document]["entity_name"] = list(mention_strs)
            major_entities[document]["entity_id"] = list(cluster_inds)
            major_entities[document]["mention_inds"] = list(mention_inds)

    return major_entities


def read_jsonl(file_name):
    data = {}
    with jsonlines.open(file_name, mode="r") as reader:
        for line in reader:
            if "doc_key" in line:
                data[line["doc_key"]] = line
    return data


def write_jsonl(file_name, data: dict):
    with jsonlines.open(file_name, mode="w") as writer:
        for key in data.keys():
            writer.write(data[key])


def process_and_save_init_results(orig_results, init_results_path):
    required_keys = ["doc_key", "clusters", "predicted_clusters", "mem", "rep_emb_list"]
    filtered_results = {}
    for document in orig_results:
        filtered_results[document] = {}
        for key in required_keys:
            if key in orig_results[document]:
                filtered_results[document][key] = orig_results[document][key]
            else:
                filtered_results[document][key] = None
    parent_directory = os.path.dirname(init_results_path)
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    # print("Filtered results:\n", filtered_results)
    write_jsonl(init_results_path, filtered_results)


def get_coref_docs(litbank_jsonlines_path):
    litbank_coref_docs = {}
    with jsonlines.open(litbank_jsonlines_path) as reader:
        for obj in reader:
            copy_dict = {}
            for key in obj:
                if key != "doc_key":
                    copy_dict[key] = obj[key]
            litbank_coref_docs[obj["doc_key"]] = copy_dict

    return litbank_coref_docs


def get_head(candidates_vs_candidatestr, candidates_vs_tbound, token_vs_tokenstr, nlp):
    candidates_vs_head_index = []
    for mention_ind, mention_str in enumerate(candidates_vs_candidatestr):
        noun_phrase = mention_str
        doc = nlp(noun_phrase)
        chunk = doc[:]
        head = chunk.root.text
        tokens = candidates_vs_tbound[mention_ind]
        token_strs = token_vs_tokenstr[tokens[0] : tokens[1] + 1]
        all_tokens = range(tokens[0], tokens[1] + 1)
        try:
            head_index = token_strs.index(head)
        except:
            for token_ind, token in enumerate(token_strs):
                if head in token:
                    head_index = token_ind
                    break

        candidates_vs_head_index.append(all_tokens[head_index])
    return candidates_vs_head_index