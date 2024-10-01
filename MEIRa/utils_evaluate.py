import os
import logging
import pickle
import time
import json
import torch
from os import path
from collections import OrderedDict, Counter

from coref_utils.metrics import CorefEvaluator, F1Evaluator
from coref_utils.conll import evaluate_conll
from coref_utils.utils import get_mention_to_cluster, is_aligned, filter_clusters

from model.utils import action_sequences_to_clusters
from model.entity_ranking_model import EntityRankingModel

from omegaconf import DictConfig
from typing import Dict
from torch import Tensor
from collections import defaultdict
import time

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger()


def get_log_file_name(
    config,
    dataset,
    teacher_force,
    gold_mentions,
    split,
    _iter,
):

    log_dir = path.join(config.paths.model_dir, dataset)

    ## Used for special experiments where we want to save logs in a different directory --
    if config.get("log_dir_add", None) is not None:
        log_dir_add = config.log_dir_add
        log_dir = path.join(log_dir, log_dir_add)

    if not path.exists(log_dir):
        os.makedirs(log_dir)

    gold_ment_str = ""
    if (
        config.model.mention_params.use_gold_ments
    ):  ## Mode where you train with golden mentions
        gold_ment_str = "_gold"

    tf_str = ""  ## Teacher forced evaluation
    if teacher_force == True:
        tf_str = "_tf"

    gold_str = ""  ## Golden mentions in evaluation
    if gold_mentions == True:
        gold_str = "_gold(eval)"

    ext_ment_str = ""  ## External mention evaluation
    if config.model.mention_params.ext_ment:
        ext_ment_str = "_ext_ment"

    log_file = path.join(
        log_dir,
        split + gold_ment_str + gold_str + tf_str + _iter + ext_ment_str + ".log.jsonl",
    )
    log_file_link = path.join(
        log_dir,
        split
        + gold_ment_str
        + gold_str
        + tf_str
        + _iter
        + ext_ment_str
        + ".link.jsonl",
    )
    print("Log file: ", log_file)
    return log_file, log_file_link


def get_logs(example, raw_predicted_clusters, coref_scores):
    log_example = dict(example)
    log_example["predicted_clusters"] = raw_predicted_clusters
    log_example["coref_scores"] = coref_scores

    del log_example["tensorized_sent"]
    for key in list(log_example.keys()):
        if isinstance(log_example[key], Tensor):
            del log_example[key]
    return log_example


def full_coref_evaluation(
    config: DictConfig,
    model: EntityRankingModel,
    data_iter_map: Dict,
    dataset: str,
    split="dev",
    _iter="",
    teacher_force=False,
    gold_mentions=False,
    final_eval=False,
    conll_data_dir: Dict = None,
) -> Dict:
    """Function to evaluate full coreference chains.

    Args:
            config: Experiment configuration
            model: Coreference model
            data_iter_map: Data iterator
            dataset: Name of the coreference dataset
            split: Partition of the dataset - train/dev/test
            final_eval: Whether this is a periodic evaluation or final evaluation
                    For final evaluation, official CoNLL scores can be calculated if possible.
            conll_data_dir:  Data directory dictionary which maps datasets to their gold CoNLL files.

    Returns:
            dict: Dictionary with results for all the metrics.
    """

    # Capture the auxiliary action accuracy
    total_actions = 0.0
    evaluator = CorefEvaluator()
    f1evaluator = F1Evaluator()
    coref_predictions, subtoken_maps = {}, {}

    logger.info(f"Evaluating on {len(data_iter_map[split][dataset])} examples")

    log_file, log_file_link = get_log_file_name(
        config,
        dataset,
        teacher_force,
        gold_mentions,
        split,
        _iter,
    )
    f = open(log_file, "w")
    f_link = open(log_file_link, "w")

    for example in data_iter_map[split][dataset]:
        ## Get outputs:
        (
            pred_mentions,
            pred_mentions_emb,
            mention_scores,
            gt_actions,
            pred_actions,
            coref_scores,
            entity_cluster_states,
            link_time,
        ) = model(example, teacher_force=teacher_force, gold_mentions=gold_mentions)

        num_major_entities = len(example["representatives"])
        raw_predicted_clusters = action_sequences_to_clusters(
            pred_actions, pred_mentions, num_major_entities
        )
        assert (
            len(raw_predicted_clusters)
            == len(example["clusters"])
            == num_major_entities + 1
        ), "Number of clusters should be equal to number of major entities + 1"

        ## Remove clusters less than the threshold of 1 and remove others from evaluation in MET here. Remove empty clustes for coref
        predicted_clusters_coref = filter_clusters(raw_predicted_clusters, threshold=1)

        ## Keep cluster numbers same as the number of major entities.
        predicted_clusters_f1 = filter_clusters(raw_predicted_clusters, threshold=0)

        ## Golden clusters cannot be empty so we can use the threshold as 1 But we remove the last cluster anyways
        gold_clusters = filter_clusters(example["clusters"], threshold=1)

        mention_to_predicted_coref = get_mention_to_cluster(predicted_clusters_coref)
        mention_to_gold = get_mention_to_cluster(gold_clusters)

        evaluator.update(
            predicted_clusters_coref,
            gold_clusters,
            mention_to_predicted_coref,
            mention_to_gold,
        )

        assert (
            len(predicted_clusters_f1) == len(gold_clusters) == num_major_entities
        ), "Predicted and Gold clusters should be of same length and equal to number of major entities + 1"

        f1evaluator.update(predicted_clusters_f1, gold_clusters)

        coref_predictions[example["doc_key"]] = raw_predicted_clusters
        if "orig_subtoken_map" in example:
            subtoken_maps[example["doc_key"]] = example["orig_subtoken_map"]
        else:
            subtoken_maps[example["doc_key"]] = example["subtoken_map"]

        total_actions += len(pred_actions)

        max_coref_scores = [max(coref_score) for coref_score in coref_scores]
        ## Removed oracle clustering for now. Code is now at the bottom  of this file.

        log_example = get_logs(
            example,
            raw_predicted_clusters=raw_predicted_clusters,
            coref_scores=max_coref_scores,
        )
        log_link_example = {
            "doc_key": example["doc_key"],
            "num_mentions": len(pred_mentions),
            "link_time": link_time,
        }
        if _iter == "":
            f.write(json.dumps(log_example) + "\n")
            f_link.write(json.dumps(log_link_example) + "\n")
    f.close()
    f_link.close()

    result_dict: Dict = OrderedDict()
    perf_str: str = ""
    # Print individual metrics
    for indv_metric, indv_evaluator in zip(config.metrics, evaluator.evaluators):
        perf_str += ", " + indv_metric + ": {}".format(indv_evaluator.get_f1() * 100)
        result_dict[indv_metric] = OrderedDict()
        result_dict[indv_metric]["recall"] = indv_evaluator.get_recall() * 100
        result_dict[indv_metric]["precision"] = indv_evaluator.get_precision() * 100
        result_dict[indv_metric]["fscore"] = indv_evaluator.get_f1() * 100

    result_dict["fscore"] = evaluator.get_f1() * 100
    result_dict["f1_macro"], result_dict["f1_micro"] = f1evaluator.get_numbers()
    logger.info("F-score: %.1f %s" % (result_dict["fscore"], perf_str))

    return result_dict


def coref_evaluation(
    config: DictConfig,
    model: EntityRankingModel,
    data_iter_map: Dict,
    dataset: str,
    split="dev",
    _iter="",
    teacher_force=False,
    gold_mentions=False,
    final_eval=False,
    conll_data_dir: Dict = None,
) -> Dict:
    """Evaluation function which calls the dataset-appropriate coreference evaluation function."""

    return full_coref_evaluation(
        config,
        model,
        data_iter_map,
        dataset,
        split=split,
        _iter=_iter,
        teacher_force=teacher_force,
        gold_mentions=gold_mentions,
        final_eval=final_eval,
        conll_data_dir=conll_data_dir,
    )
