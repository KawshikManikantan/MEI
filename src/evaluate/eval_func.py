import os
from collections import OrderedDict

from utils.metrics import CorefEvaluator
from typing import Dict
import jsonlines
import pandas as pd

from utils.utils import get_major_entities


def process_performances(list_doc_performances):
    list_doc_peformances_processed = []
    for document in list_doc_performances:
        processed_dict = {}
        for metric in document:
            if type(document[metric]) == OrderedDict:
                for key in document[metric]:
                    processed_dict[metric + "_" + key] = document[metric][key]
            else:
                processed_dict[metric] = document[metric]
        list_doc_peformances_processed.append(processed_dict)
    return list_doc_peformances_processed


def get_mention_to_cluster(clusters) -> Dict:
    """Get mention to cluster mapping."""

    clusters = [tuple(tuple(mention) for mention in cluster) for cluster in clusters]
    mention_to_cluster_dict = {}
    for cluster in clusters:
        for mention in cluster:
            mention_to_cluster_dict[mention] = cluster
    return mention_to_cluster_dict


def full_coref_evaluation(
    config,
    predicted_clusters_coref,
    golden_clusters_coref,
):
    evaluator = CorefEvaluator()
    list_doc_performances = []
    for document in predicted_clusters_coref:
        evaluator_perdoc = CorefEvaluator()

        predicted_clusters = predicted_clusters_coref[document]
        mention_to_predicted = get_mention_to_cluster(predicted_clusters)

        gold_clusters = golden_clusters_coref[document]
        mention_to_gold = get_mention_to_cluster(gold_clusters)

        evaluator.update(
            predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold
        )

        evaluator_perdoc.update(
            predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold
        )

        result_dict: Dict = OrderedDict()

        # Print individual metrics
        for indv_metric, indv_evaluator in zip(
            config["metrics"], evaluator_perdoc.evaluators
        ):
            result_dict[indv_metric] = OrderedDict()
            result_dict[indv_metric]["recall"] = indv_evaluator.get_recall() * 100
            result_dict[indv_metric]["precision"] = indv_evaluator.get_precision() * 100
            result_dict[indv_metric]["fscore"] = indv_evaluator.get_f1() * 100

        result_dict["fscore"] = evaluator_perdoc.get_f1() * 100
        result_dict["doc_key"] = document
        list_doc_performances.append(result_dict)

    result_dict: Dict = OrderedDict()

    # Print individual metrics
    for indv_metric, indv_evaluator in zip(config["metrics"], evaluator.evaluators):
        result_dict[indv_metric] = OrderedDict()
        result_dict[indv_metric]["recall"] = indv_evaluator.get_recall() * 100
        result_dict[indv_metric]["precision"] = indv_evaluator.get_precision() * 100
        result_dict[indv_metric]["fscore"] = indv_evaluator.get_f1() * 100

    result_dict["fscore"] = evaluator.get_f1() * 100

    return result_dict, list_doc_performances


def get_f1(
    predicted_clusters_f1,
    golden_clusters_f1,
    major_entities,
    list_doc_performances_processed,
):

    f1_macro = 0.0
    f1_micro = 0.0
    f1_macro_micro = 0.0

    macro_support = 0
    micro_support = 0
    per_entity_stats = {
        "doc_name": [],
        "entity_name": [],
        "f1_score": [],
        "support": [],
    }

    for doc_ind, document in enumerate(list_doc_performances_processed):
        doc_name = document["doc_key"]
        print(doc_name)
        predicted = predicted_clusters_f1[doc_name]
        gold = golden_clusters_f1[doc_name]
        key_entities = major_entities[doc_name]["entity_name"]
        total_entities = key_entities + ["Others"]
        f1_macro_doc = 0.0
        f1_micro_doc = 0.0
        macro_support_doc = 0
        micro_support_doc = 0
        assert (
            len(gold) == len(key_entities) + 1
        ), "Number of clusters in gold and key entities mismatch"
        for cluster_ind, cluster in enumerate(gold[:-1]):
            predicted_set = (
                set(predicted[cluster_ind]) if cluster_ind < len(predicted) else set()
            )
            correct = set(cluster).intersection(set(predicted_set))
            num_correct = len(correct)
            num_predicted = len(predicted_set)
            num_gt = len(cluster)
            precision = num_correct / num_predicted if num_predicted > 0 else 0
            recall = num_correct / num_gt if num_gt > 0 else 0
            f1_score = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0
            )
            support_entity_micro = num_gt
            support_entity_macro = 1
            print("F1_score: ", f1_score, "Support: ", support_entity_micro)

            f1_macro += f1_score * support_entity_macro
            f1_micro += f1_score * support_entity_micro
            f1_macro_doc += f1_score * support_entity_macro
            f1_micro_doc += f1_score * support_entity_micro
            macro_support += support_entity_macro
            micro_support += support_entity_micro
            macro_support_doc += support_entity_macro
            micro_support_doc += support_entity_micro

            per_entity_stats["doc_name"].append(doc_name)
            per_entity_stats["entity_name"].append(total_entities[cluster_ind])
            per_entity_stats["f1_score"].append(f1_score)
            per_entity_stats["support"].append(support_entity_micro)

        f1_macro_doc = f1_macro_doc / macro_support_doc if macro_support_doc > 0 else 0
        f1_micro_doc = f1_micro_doc / micro_support_doc if micro_support_doc > 0 else 0
        list_doc_performances_processed[doc_ind]["f1_macro"] = round(
            f1_macro_doc * 100, 2
        )
        list_doc_performances_processed[doc_ind]["f1_micro"] = round(
            f1_micro_doc * 100, 2
        )

        f1_macro_micro += f1_micro_doc
    macro_f1 = f1_macro / macro_support * 100
    micro_f1 = f1_micro / micro_support * 100
    macro_micro_f1 = f1_macro_micro / len(list_doc_performances_processed) * 100
    print("Macro F1:", macro_f1)
    print("Micro F1:", micro_f1)
    print("Macro-Micro F1:", macro_micro_f1)
    return (
        macro_f1,
        micro_f1,
        macro_micro_f1,
        list_doc_performances_processed,
        per_entity_stats,
    )


def process_clusters_coref(predicted_clusters, golden_clusters, major_entities):
    predicted_clusters_coref = {}
    golden_clusters_coref = {}
    for doc_name in predicted_clusters:
        major_entities_doc = major_entities[doc_name]["entity_name"]
        num_classes = len(major_entities_doc)
        golden_clusters_coref[doc_name] = golden_clusters[doc_name][:num_classes]
        predicted_clusters_coref[doc_name] = predicted_clusters[doc_name][:num_classes]

    return predicted_clusters_coref, golden_clusters_coref


def process_clusters_f1(predicted_clusters, golden_clusters, major_entities):
    predicted_clusters_f1 = {}
    golden_clusters_f1 = {}
    for doc_name in predicted_clusters:
        major_entities_doc = major_entities[doc_name]["entity_name"]
        num_classes = len(major_entities_doc) + 1

        ## Ignored the failed mentions here: The failed mentions are automatically added to the others class:
        predicted_clusters_f1_doc = predicted_clusters[doc_name][:num_classes]
        predicted_mentions = sorted(
            [mention for cluster in predicted_clusters_f1_doc for mention in cluster]
        )

        golden_clusters_f1_doc = golden_clusters[doc_name][:num_classes]
        golden_mentions = sorted(
            [mention for cluster in golden_clusters_f1_doc for mention in cluster]
        )

        if len(golden_clusters_f1_doc) < num_classes:
            golden_clusters_f1_doc.append([])

        if len(predicted_clusters_f1_doc) < num_classes:
            predicted_clusters_f1_doc.append([])

        new_gold_mentions = 0
        for mention in predicted_mentions:
            if mention not in golden_mentions:
                golden_clusters_f1_doc[-1].append(mention)
                new_gold_mentions += 1

        new_pred_mentions = 0
        for mention in golden_mentions:
            if mention not in predicted_mentions:
                predicted_clusters_f1_doc[-1].append(mention)
                new_pred_mentions += 1

        predicted_clusters_f1[doc_name] = predicted_clusters_f1_doc
        golden_clusters_f1[doc_name] = golden_clusters_f1_doc

    return predicted_clusters_f1, golden_clusters_f1


def evaluate(experiment_config):
    dataset_configs = experiment_config["datasets"]
    doc_split = experiment_config["split"]
    eval_mode = experiment_config["eval"]
    for dataset in dataset_configs:
        doc_me = dataset_configs[dataset][f"{doc_split}_me"]
        major_entities = get_major_entities(doc_me)

        dataset_config = {
            "cluster_threshold": 1,
            "canonical_cluster_threshold": 1,
            "metrics": ["MUC", "Bcub", "CEAFE"],
        }

        result_destination = experiment_config["paths"]["result_destination"]
        result_jsonl_file = os.path.join(
            result_destination, dataset, doc_split, "result.jsonl"
        )

        file_name = os.path.splitext(result_jsonl_file)[0]
        xlsx_file = f"{file_name}.xlsx"
        xlsx_ent_file = f"{file_name}_per_entity.xlsx"
        txt_file = f"{file_name}_avg.txt"

        # print("Jsonl file: ", result_jsonl_file)

        pred_outputs = {}
        with jsonlines.open(result_jsonl_file, mode="r") as reader:
            for line in reader:
                pred_outputs[line["doc_key"]] = line

        predicted_clusters = {}
        golden_clusters = {}
        for document in pred_outputs:
            if pred_outputs[document].get("golden_clusters") is None:
                pred_outputs[document]["golden_clusters"] = pred_outputs[document][
                    "clusters"
                ]

            eval_golden_clusters = pred_outputs[document]["golden_clusters"]
            eval_predicted_clusters = pred_outputs[document]["predicted_clusters"]

            if eval_mode == "head":
                eval_golden_clusters = pred_outputs[document]["golden_clusters_head"]
                eval_predicted_clusters = pred_outputs[document][
                    "predicted_clusters_head"
                ]
                assert (
                    eval_golden_clusters is not None
                    and eval_predicted_clusters is not None
                ), "Head mode activated but clusters unavailable."

            predicted_clusters[document] = [
                [tuple(mention) for mention in cluster]
                for cluster in eval_predicted_clusters
            ]
            golden_clusters[document] = [
                [tuple(mention) for mention in cluster]
                for cluster in eval_golden_clusters
            ]

        predicted_clusters_coref, golden_clusters_coref = process_clusters_coref(
            predicted_clusters, golden_clusters, major_entities=major_entities
        )
        result_dict, list_doc_performances = full_coref_evaluation(
            dataset_config, predicted_clusters_coref, golden_clusters_coref
        )

        list_doc_performances_processed = process_performances(list_doc_performances)

        predicted_clusters_f1, golden_clusters_f1 = process_clusters_f1(
            predicted_clusters, golden_clusters, major_entities
        )
        (
            macro_f1,
            micro_f1,
            macro_micro_f1,
            list_doc_performances_processed,
            per_entity_stats,
        ) = get_f1(
            predicted_clusters_f1,
            golden_clusters_f1,
            major_entities,
            list_doc_performances_processed,
        )

        doc_perfomances = pd.DataFrame(list_doc_performances_processed)
        doc_perfomances.to_excel(xlsx_file)

        per_entity_stats = pd.DataFrame(per_entity_stats)
        per_entity_stats.to_excel(xlsx_ent_file)

        avg_performance_txt = f"""
            "CONLL F1": {result_dict["fscore"]}
            "Macro F1": {macro_f1}
            "Micro F1": {micro_f1}
            "Macro-Micro F1": {macro_micro_f1}
        """
        print(avg_performance_txt)
        with open(txt_file, "w") as f:
            f.write(avg_performance_txt)
