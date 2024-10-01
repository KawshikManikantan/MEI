import os
from os import path
from collections import defaultdict
import jsonlines


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


def get_major_entities(input_jsonlines):
    if path.isdir(input_jsonlines):
        doc_names = [
            file_name[:-6]
            for file_name in os.listdir(input_jsonlines)
            if file_name.endswith(".jsonl")
        ]
        major_entities = {}
        for document in doc_names:
            print(document)
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
            mention_inds, mention_strs, cluster_inds = zip(
                *sorted(zip(mention_inds, mention_strs, cluster_inds))
            )
            major_entities[document]["entity_name"] = list(mention_strs)
            major_entities[document]["entity_id"] = list(cluster_inds)
            print(mention_inds)
            print(mention_strs)

    return major_entities


def get_modified_clusters(clusters_init, major_entities_doc, prompt_type):
    clusters = []
    if prompt_type == "2_b":
        for entity_ind in major_entities_doc["entity_id"]:
            clusters.append(clusters_init[entity_ind])
        for cluster_ind, cluster in enumerate(clusters_init):
            if cluster_ind not in major_entities_doc["entity_id"]:
                clusters.append(cluster)
    elif prompt_type == "1":
        for entity_ind in major_entities_doc["entity_id"]:
            clusters.append(clusters_init[entity_ind])
    else:
        for entity_ind in major_entities_doc["entity_id"]:
            clusters.append(clusters_init[entity_ind])
        others = []
        for cluster_ind, cluster in enumerate(clusters_init):
            if cluster_ind not in major_entities_doc["entity_id"]:
                others.extend(cluster)
        clusters.append(others)

    clusters = [sorted(cluster) for cluster in clusters]

    return clusters


def read_jsonl_lst(file_name):
    data = []
    with jsonlines.open(file_name, mode="r") as reader:
        for line in reader:
            data.append(line)
    return data


def read_jsonl(file_name):
    data = {}
    with jsonlines.open(file_name, mode="r") as reader:
        for line in reader:
            if "doc_key" in line:
                data[line["doc_key"]] = line
    return data


def write_jsonl(file_name, data):
    with jsonlines.open(file_name, mode="w") as writer:
        for key in data.keys():
            writer.write(data[key])


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
