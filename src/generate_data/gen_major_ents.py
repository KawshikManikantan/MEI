import os
import jsonlines
import hydra
import numpy as np
import spacy

from tqdm.auto import tqdm
from collections import Counter

from utils.get_processed_dataset import get_coref_docs, get_processed_dataset

from gen_mei_data.config import *


def generate_max_entities_details(coref_docs_processed, tsv_folder):
    nlp = spacy.load("en_core_web_trf")
    major_entities = {}
    for document in tqdm(coref_docs_processed.keys()):
        mentions = coref_docs_processed[document]["mentions_vs_stbound"]
        mentions_str = coref_docs_processed[document]["mentions_vs_mentionstr"]
        cluster_lens = np.array(
            [
                len(cluster)
                for cluster in coref_docs_processed[document]["clusters_vs_stbound"]
            ]
        )
        cluster_max_inds = np.argsort(cluster_lens)[::-1]
        max_clusters_doc = []

        mask_info = []
        if tsv_folder is not None:
            for ctgry in coref_docs_processed[document]["mentions_vs_mentionctgry"]:
                if ctgry == "PRON":
                    mask_info.append(0)
                elif ctgry == "PROP":
                    mask_info.append(1)
                else:
                    mask_info.append(2)
        else:
            for mention_ind in range(len(mentions_str)):
                if mentions_str[mention_ind].lower() in PRONOUNS_GROUPS:
                    mask_info.append(0)
                elif nlp(mentions_str[mention_ind]).ents != ():
                    mask_info.append(1)
                else:
                    mask_info.append(2)

        for cluster_ind in cluster_max_inds:
            if cluster_lens[cluster_ind] < MIN_MENTIONS:
                break
            if len(max_clusters_doc) == MAX_ENTITIES:
                break

            cluster = coref_docs_processed[document]["clusters_vs_stbound"][cluster_ind]
            num_mentions_cluster = len(cluster)
            cluster_mention_ind = [mentions.index(mention) for mention in cluster]

            # print(len(cluster_mention_ind),len(mentions_str),len(mask_info))
            cluster_pron = [
                mentions_str[mention_ind].lower()
                for mention_ind in cluster_mention_ind
                if mask_info[mention_ind] == 0
            ]
            cluster_names = [
                mentions_str[mention_ind].lower()
                for mention_ind in cluster_mention_ind
                if mask_info[mention_ind] == 1
            ]
            cluster_nominals = [
                mentions_str[mention_ind].lower()
                for mention_ind in cluster_mention_ind
                if mask_info[mention_ind] == 2
            ]

            counter_name = Counter(cluster_names)
            counter_nom = Counter(cluster_nominals)
            counter_pron = Counter(cluster_pron)

            name_str = "".join(
                f"{key}_{counter_name[key]}\n" for key in counter_name.keys()
            )
            nom_str = "".join(
                f"{key}_{counter_nom[key]}\n" for key in counter_nom.keys()
            )
            pron_str = "".join(
                f"{key}_{counter_pron[key]}\n" for key in counter_pron.keys()
            )

            if (
                len(cluster_names) < MIN_NAMES_NUM
                and len(cluster_names) / num_mentions_cluster < MIN_NAMES_PERC
            ):
                if (
                    len(cluster_nominals) < MIN_NOMINALS_NUM
                    and len(cluster_nominals) / num_mentions_cluster < MIN_NOMINALS_PERC
                ):
                    counter_pron_max = counter_pron.most_common(1)
                    if counter_pron_max:
                        cluster_pron_max_ele = counter_pron_max[0][0]
                        if (
                            cluster_pron_max_ele == "i"
                            and len(cluster_pron) / num_mentions_cluster > MIN_I_PERC
                        ):
                            mention_str_cluster = "i"
                        else:
                            continue
                else:
                    mention_str_cluster = counter_nom.most_common(1)[0][0]
            else:
                mention_str_cluster = counter_name.most_common(1)[0][0]

            cluster_mention_str = [
                mentions_str[mention_ind].lower() for mention_ind in cluster_mention_ind
            ]  ## All mentions in the cluster
            mention_selected_ind = cluster_mention_ind[
                cluster_mention_str.index(mention_str_cluster)
            ]  ## Index of the selected mention in the document mention
            max_clusters_doc.append(
                {
                    "cluster_ind": int(cluster_ind),
                    "mention_ind": mention_selected_ind,
                    "mention_count": num_mentions_cluster,
                    "mention_str": mention_str_cluster,
                    "name_str": name_str,
                    "nom_str": nom_str,
                    "pron_str": pron_str,
                }
            )
        major_entities[document] = max_clusters_doc
    return major_entities


def write_major_entities(major_entities, destination):
    with jsonlines.open(destination, mode="w") as writer:
        for document in major_entities.keys():
            cluster_inds = [
                entity["cluster_ind"] for entity in major_entities[document]
            ]
            mention_inds = [
                entity["mention_ind"] for entity in major_entities[document]
            ]
            mention_strs = [
                entity["mention_str"] for entity in major_entities[document]
            ]
            writer.write(
                {
                    "doc_key": document,
                    "cluster_inds": cluster_inds,
                    "mention_inds": mention_inds,
                    "mention_strs": mention_strs,
                }
            )


def process_split(split_file, tsv_folder, destination):
    docs = get_coref_docs(split_file)
    docs_processed = get_processed_dataset(docs, tsv_folder)
    major_entities = generate_max_entities_details(docs_processed, tsv_folder)
    write_major_entities(major_entities, destination)


@hydra.main(config_path=f"{os.getcwd()}/configs/args", config_name="args")
def main(args):
    for dataset in args["datasets"]:

        tsv_folder = args["datasets"][dataset].get("tsv", None)

        ## Get the train, dev and test files
        train_file = args["datasets"][dataset].get("train_file", None)

        if dataset != "litbank":
            dev_file = args["datasets"][dataset].get("dev_file", None)
        else:
            dev_file = None

        if dataset not in ["litbank", "fantasy"]:
            test_file = args["datasets"][dataset].get("test_file", None)
        else:
            test_file = None

        ## Get the destination file
        train_me = args["datasets"][dataset].get("train_me", None)
        dev_me = args["datasets"][dataset].get("dev_me", None)
        test_me = args["datasets"][dataset].get("test_me", None)

        if train_file:
            process_split(train_file, tsv_folder, train_me)

        if dev_file:
            process_split(dev_file, tsv_folder, dev_me)

        if test_file:
            process_split(test_file, tsv_folder, test_me)


if __name__ == "__main__":
    main()
