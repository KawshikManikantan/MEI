import jsonlines
import os
import spacy
from utils.utils import get_head


def get_coref_docs(jsonlines_path):
    coref_docs = {}
    with jsonlines.open(jsonlines_path) as reader:
        for obj in reader:
            copy_dict = {}
            for key in obj:
                if key != "doc_key":
                    copy_dict[key] = obj[key]
            coref_docs[obj["doc_key"]] = copy_dict

    return coref_docs


def docs_to_jsonlines(coref_docs, path_to_jsonlines):
    doc_list = []
    for doc_key in coref_docs:
        doc_dict = {}
        doc_dict["doc_key"] = doc_key
        for key in coref_docs[doc_key]:
            doc_dict[key] = coref_docs[doc_key][key]
        doc_list.append(doc_dict)
    with jsonlines.open(path_to_jsonlines, mode="w") as writer:
        writer.write_all(doc_list)


def get_sys_output(eval_file):
    sys_output = {}
    # keys_tracked = ['pred_mentions', 'pred_mentions_emb', 'mention_scores', 'pred_actions', 'predicted_clusters', 'coref_scores','gt_actions']
    # keys_tracked =
    with jsonlines.open(eval_file) as reader:
        for obj in reader:
            copy_dict = {}
            for key in obj:
                # if key in obj:
                copy_dict[key] = obj[key]
            sys_output[obj["doc_key"]] = copy_dict

    return sys_output


def merge_tokens(lst):
    """Merges token between the speaker_Start and speaker_End"""
    merged_list = []
    merge_flag = False
    current_speaker = ""

    for token in lst:
        if token == "[SPEAKER_START]":
            merge_flag = True
            current_speaker = "[SPEAKER_START]"
        elif token == "[SPEAKER_END]":
            merge_flag = False
            current_speaker += token
            merged_list.append(current_speaker)
        elif merge_flag:
            current_speaker += token
        else:
            merged_list.append(token)

    return merged_list


def higher_to_lower(subtoken_map):
    "Converts lower to higher map into higher to lower map"
    max_tokens = subtoken_map[-1]
    token_map = [[] for _ in range(max_tokens + 1)]
    for subtoken, token in enumerate(subtoken_map):
        token_map[token].append(subtoken)
    return token_map


def sbounds_to_tbounds(sbounds, stot):
    "Convert subtoken bounds to token bounds"
    tbounds = []
    for bound in sbounds:
        tbounds.append([stot[limit] for limit in bound])
    return tbounds


def mentions_sbounds(clusters):
    """Defining Mentions from clusters"""
    return sorted([mention for cluster in clusters for mention in cluster])


def mention_to_cluster(clusters, mentions_vs_ind):
    total_mentions = len(list(mentions_vs_ind.keys()))
    mention_vs_cluster = [0 for _ in range(total_mentions)]
    for ind, cluster in enumerate(clusters):
        for mention in cluster:
            mention_vs_cluster[mentions_vs_ind[tuple(mention)]] = ind
    return mention_vs_cluster


def get_mention_type_category(tsv_litbank):
    document_list = os.listdir(tsv_litbank)
    doc_names = []

    for doc in document_list:
        doc_name = doc.split(".")[0]
        if doc_name not in doc_names:
            doc_names.append(f"{doc_name}_0")

    tokens_dict = {}
    tokens_len_dict = {}
    mention_type_dict = {}
    mention_category_dict = {}
    mention_text_dict = {}

    for doc_name in doc_names:
        token_doc = []
        len_token_doc = [0]
        with open(os.path.join(tsv_litbank, f"{doc_name[:-2]}.txt")) as fp:
            for line in fp:
                tokens = line.split()
                token_doc += tokens
                len_token_doc.append(len(tokens) + len_token_doc[-1])

        tokens_dict[doc_name] = token_doc
        tokens_len_dict[doc_name] = len_token_doc

    # print(tokens_dict)
    # print(tokens_len_dict)

    for doc_name in doc_names:
        mention_type_dict_ind = {}
        mention_category_dict_ind = {}
        mention_text_dict_ind = {}
        with open(os.path.join(tsv_litbank, f"{doc_name[:-2]}.ann")) as fp:
            for line in fp:
                line = line.strip()
                line_toks = line.split("\t")
                # print(line_toks)
                if line_toks[0] == "MENTION":
                    if line_toks[2] != line_toks[4]:
                        print("Should not happen")
                    line_num = int(line_toks[2])
                    start_tok = int(line_toks[3])
                    end_tok = int(line_toks[5])
                    text_tok = line_toks[6]
                    mention_type = line_toks[7]
                    mention_category = line_toks[8]
                    mention_token_bound = (
                        tokens_len_dict[doc_name][line_num] + start_tok,
                        tokens_len_dict[doc_name][line_num] + end_tok,
                    )
                    mention_type_dict_ind[mention_token_bound] = mention_type
                    mention_category_dict_ind[mention_token_bound] = mention_category
                    mention_text_dict_ind[mention_token_bound] = text_tok

            mention_category_dict[doc_name] = mention_category_dict_ind
            mention_type_dict[doc_name] = mention_type_dict_ind
            mention_text_dict[doc_name] = mention_text_dict_ind
    return mention_type_dict, mention_category_dict, mention_text_dict


def get_segment(doc, mentions):
    doc_boundaries = [0]
    for segment in doc:
        doc_boundaries.append(doc_boundaries[-1] + len(segment))
    mentions_vs_segments = []
    # print(doc_boundaries)
    for mention in mentions:
        # print(mention)
        for boundary_ind in range(len(doc_boundaries)):
            if mention[-1] < doc_boundaries[boundary_ind]:
                break
        mentions_vs_segments.append(boundary_ind - 1)
    # print(mentions_vs_segments)
    return mentions_vs_segments


def get_processed_dataset(coref_docs, tsv_litbank, head=False):

    nlp = spacy.load("en_core_web_trf")

    coref_docs_proc = {}
    for doc_key, doc in coref_docs.items():
        # print(doc_key)
        subtoken_vs_token = doc["subtoken_map"]
        clusters_vs_stbound = sorted([sorted(cluster) for cluster in doc["clusters"]])
        clusters_vs_tbound = [
            sbounds_to_tbounds(cluster, subtoken_vs_token)
            for cluster in clusters_vs_stbound
        ]
        subtoken_vs_sentence = doc["sentence_map"]
        token_vs_tokenstr = merge_tokens(doc["orig_tokens"])
        mentions_vs_stbound = mentions_sbounds(clusters_vs_stbound)
        mentions_vs_tbound = sbounds_to_tbounds(mentions_vs_stbound, subtoken_vs_token)
        token_vs_subtoken = higher_to_lower(subtoken_vs_token)
        sentence_vs_subtoken = higher_to_lower(subtoken_vs_sentence)
        token_vs_sentence = [
            subtoken_vs_sentence[token[0]] for token in token_vs_subtoken
        ]
        mentions_vs_sentence = [
            subtoken_vs_sentence[mention[0]] for mention in mentions_vs_stbound
        ]
        sentence_vs_subtoken = higher_to_lower(subtoken_vs_sentence)
        sentence_vs_token = higher_to_lower(token_vs_sentence)
        sentence_vs_sentencestr = "\n".join(
            [
                " ".join([token_vs_tokenstr[token] for token in tok_sentence])
                for tok_sentence in sentence_vs_token
            ]
        )
        stbound_wrt_mentions = {
            tuple(value): ind for ind, value in enumerate(mentions_vs_stbound)
        }
        mentions_vs_mentionstr = [
            " ".join(token_vs_tokenstr[tokenbound[0] : tokenbound[1] + 1])
            for tokenbound in mentions_vs_tbound
        ]
        clusters_vs_mentions = [
            [stbound_wrt_mentions[tuple(mention)] for mention in cluster]
            for cluster in clusters_vs_stbound
        ]
        mentions_vs_clusters = mention_to_cluster(
            clusters_vs_stbound, stbound_wrt_mentions
        )
        mentions_vs_segments = get_segment(doc["sentences"], mentions_vs_stbound)

        if head:
            mentions_vs_head = get_head(
                mentions_vs_mentionstr, mentions_vs_tbound, token_vs_tokenstr, nlp
            )
            clusters_vs_headbound = [
                [
                    [mentions_vs_head[mention], mentions_vs_head[mention]]
                    for mention in cluster
                ]
                for cluster in clusters_vs_mentions
            ]
            mentions_vs_headbound = [[token, token] for token in mentions_vs_head]
            clusters_vs_headstbound = [
                [
                    [mentions_vs_head[mention], mentions_vs_head[mention]]
                    for mention in cluster
                ]
                for cluster in clusters_vs_mentions
            ]
        else:
            mentions_vs_head = []
            clusters_vs_headbound = []
            mentions_vs_headbound = []
            clusters_vs_headstbound = []

        coref_docs_proc[doc_key] = {
            "subtoken_vs_token": subtoken_vs_token,
            "clusters_vs_tbound": clusters_vs_tbound,
            "clusters_vs_stbound": clusters_vs_stbound,
            "subtoken_vs_sentence": subtoken_vs_sentence,
            "token_vs_tokenstr": token_vs_tokenstr,
            "mentions_vs_stbound": mentions_vs_stbound,
            "mentions_vs_tbound": mentions_vs_tbound,
            "token_vs_subtoken": token_vs_subtoken,
            "sentence_vs_subtoken": sentence_vs_subtoken,
            "token_vs_sentence": token_vs_sentence,
            "mentions_vs_sentence": mentions_vs_sentence,
            "sentence_vs_subtoken": sentence_vs_subtoken,
            "sentence_vs_token": sentence_vs_token,
            "sentence_vs_sentencestr": sentence_vs_sentencestr,
            "stbound_wrt_mentions": stbound_wrt_mentions,
            "mentions_vs_mentionstr": mentions_vs_mentionstr,
            "clusters_vs_mentions": clusters_vs_mentions,
            "mentions_vs_clusters": mentions_vs_clusters,
            "mentions_vs_segments": mentions_vs_segments,
            "mentions_vs_head": mentions_vs_head,
            "mentions_vs_headbound": mentions_vs_headbound,
            "clusters_vs_headbound": clusters_vs_headbound,
            "clusters_vs_headstbound": clusters_vs_headstbound,
        }

    if tsv_litbank is not None:
        mention_type_dict, mention_category_dict, mention_text_dict = (
            get_mention_type_category(tsv_litbank)
        )
        for doc_name in coref_docs_proc:
            type_list = [
                mention_type_dict[doc_name][tuple(bound)]
                for bound in coref_docs_proc[doc_name]["mentions_vs_tbound"]
            ]
            category_list = [
                mention_category_dict[doc_name][tuple(bound)]
                for bound in coref_docs_proc[doc_name]["mentions_vs_tbound"]
            ]
            coref_docs_proc[doc_name]["mentions_vs_mentiontype"] = type_list
            coref_docs_proc[doc_name]["mentions_vs_mentionctgry"] = category_list
            coref_docs_proc[doc_name]["mentiontype_wrt_tbound"] = {
                key: value
                for key, value in mention_type_dict[doc_name].items()
                if list(key) in coref_docs_proc[doc_name]["mentions_vs_tbound"]
            }
            coref_docs_proc[doc_name]["mentionctgry_wrt_tbound"] = {
                key: value
                for key, value in mention_type_dict[doc_name].items()
                if list(key) in coref_docs_proc[doc_name]["mentions_vs_tbound"]
            }

    return coref_docs_proc
