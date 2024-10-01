from utils.get_processed_dataset import *
from utils.utils import *
from src.configs.config import *


def get_mention_str_tbound(mention_tbound, doc_processed_doc):
    token_vs_tokenstr = doc_processed_doc["token_vs_tokenstr"]
    mention_str = " ".join(
        [
            token_vs_tokenstr[token_ind]
            for token_ind in range(mention_tbound[0], mention_tbound[1] + 1)
        ]
    )
    return mention_str


def fill_sys_output(decision_dict, sys_output, major_entities):
    for document in sys_output.keys():
        major_entity_names = major_entities[document]["entity_name"] + ["Others"]

        ### Failed mentions
        if (
            len(sys_output[document]["predicted_clusters"])
            <= len(sys_output[document]["golden_clusters"]) + 1
        ):
            failed_mentions = []
        else:
            failed_mentions = sys_output[document]["predicted_clusters"][-1]

        for mention in failed_mentions:
            decision_dict["doc_key"].append(document)
            decision_dict["mention_start"].append(mention[0])
            decision_dict["mention_end"].append(mention[1])
            decision_dict["broad_mention_type"].append("Failed")
            is_golden_mention = (
                1 if mention in sys_output[document]["gt_mentions"] else 0
            )
            decision_dict["golden_mention"].append(is_golden_mention)
            if is_golden_mention:
                entity_ind = sys_output[document]["gt_mentions_vs_clusters"][
                    tuple(mention)
                ]
                entity_name = major_entity_names[entity_ind]
            else:
                entity_ind = -1
                entity_name = "None"
            decision_dict["entity_ind"].append(entity_ind)
            decision_dict["entity_name"].append(entity_name)
            decision_dict["mapped_entitiy_ind"].append(-1)
            decision_dict["mapped_entity_name"].append("None")

        ### Predicted mentions
        if (
            len(sys_output[document]["predicted_clusters"])
            <= len(sys_output[document]["golden_clusters"]) + 1
        ):
            predicted_clusters_proper = sys_output[document]["predicted_clusters"]
        else:
            predicted_clusters_proper = sys_output[document]["predicted_clusters"][:-1]
        predicted_mentions_proper = sorted(
            [mention for cluster in predicted_clusters_proper for mention in cluster]
        )

        for mention in predicted_mentions_proper:
            decision_dict["doc_key"].append(document)
            decision_dict["mention_start"].append(mention[0])
            decision_dict["mention_end"].append(mention[1])
            decision_dict["broad_mention_type"].append("Predicted")
            is_golden_mention = (
                1 if mention in sys_output[document]["gt_mentions"] else 0
            )
            decision_dict["golden_mention"].append(is_golden_mention)
            if is_golden_mention:
                entity_ind = sys_output[document]["gt_mentions_vs_clusters"][
                    tuple(mention)
                ]
                entity_name = major_entity_names[entity_ind]
            else:
                entity_ind = -1
                entity_name = "None"
            decision_dict["entity_ind"].append(entity_ind)
            decision_dict["entity_name"].append(entity_name)
            mapped_entity_ind = sys_output[document]["predicted_mentions_vs_clusters"][
                tuple(mention)
            ]
            mapped_entity_name = major_entity_names[mapped_entity_ind]
            decision_dict["mapped_entitiy_ind"].append(mapped_entity_ind)
            decision_dict["mapped_entity_name"].append(mapped_entity_name)

        ### Missed Mentions
        golden_mentions = sys_output[document]["gt_mentions"]
        predicted_mentions = sys_output[document]["predicted_mentions"]
        for mention_ind, mention in enumerate(golden_mentions):
            if mention not in predicted_mentions:
                decision_dict["doc_key"].append(document)
                decision_dict["mention_start"].append(mention[0])
                decision_dict["mention_end"].append(mention[1])
                decision_dict["broad_mention_type"].append("Missed")
                decision_dict["golden_mention"].append(1)
                entity_ind = sys_output[document]["gt_mentions_vs_clusters"][
                    tuple(mention)
                ]
                entity_name = major_entity_names[entity_ind]
                decision_dict["entity_ind"].append(entity_ind)
                decision_dict["entity_name"].append(entity_name)
                decision_dict["mapped_entitiy_ind"].append(-1)
                decision_dict["mapped_entity_name"].append("None")


def get_mention_dets(
    decision_dict,
    sys_output,
    docs_processed,
    major_entities,
    original_index_list_key,
    doc_tsv_addr,
):
    for ind in range(len(decision_dict["doc_key"])):
        mention = [
            decision_dict["mention_start"][ind],
            decision_dict["mention_end"][ind],
        ]
        document = decision_dict["doc_key"][ind]
        decision_dict["mention_str"].append(
            get_mention_str_tbound(mention, docs_processed[document])
        )
        if decision_dict["golden_mention"][ind] == 1:
            mention_ind_gold_orig = docs_processed[document][
                original_index_list_key
            ].index(mention)
            decision_dict["mention_ind"].append(mention_ind_gold_orig)
            if doc_tsv_addr is not None:
                decision_dict["category"].append(
                    docs_processed[document]["mentions_vs_mentionctgry"][
                        mention_ind_gold_orig
                    ]
                )
            else:
                if decision_dict["mention_str"][-1].lower().strip() in PRONOUNS_GROUPS:
                    decision_dict["category"].append("PRON")
                else:
                    decision_dict["category"].append("NOM")
            decision_dict["nested_mention"].append(
                sys_output[document]["gt_mentions_nested"][mention_ind_gold_orig]
            )
        else:
            decision_dict["mention_ind"].append(-1)
            decision_dict["category"].append("None")
            predicted_mention_ind = sys_output[document]["predicted_mentions"].index(
                mention
            )
            decision_dict["nested_mention"].append(
                sys_output[document]["predicted_mentions_nested"][predicted_mention_ind]
            )
