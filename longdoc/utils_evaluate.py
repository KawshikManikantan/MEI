import os
import logging
import pickle
import time
import json
import torch
from os import path
from collections import OrderedDict, Counter

from coref_utils.metrics import CorefEvaluator
from coref_utils.utils import get_mention_to_cluster, is_aligned, filter_clusters

from model.utils import action_sequences_to_clusters
from model.entity_ranking_model import EntityRankingModel

from omegaconf import DictConfig
from typing import Dict
from torch import Tensor
from collections import defaultdict

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

    if not path.exists(log_dir):
        os.makedirs(log_dir)

    gold_ment_str = ""
    if (
        config.model.mention_params.use_gold_ments
    ):  ## Model is trained with gold mentions
        gold_ment_str = "_gold"

    tf_str = ""
    if teacher_force == True:  ## Teacher forcing during evaluation
        tf_str = "_tf"

    gold_str = ""
    if gold_mentions == True:  ## Golden mentions during evaluation
        gold_str = "_gold(eval)"

    ext_ment_str = ""  ## External mentions during evaluation
    if config.model.mention_params.ext_ment:
        ext_ment_str = "_ext_ment"

    log_file = path.join(
        log_dir,
        split + gold_ment_str + gold_str + tf_str + _iter + ext_ment_str + ".log.jsonl",
    )

    log_link_file = path.join(
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
    return log_dir, log_file, log_link_file


def get_logs(example, raw_predicted_clusters):
    log_example = dict(example)
    log_example["predicted_clusters"] = raw_predicted_clusters
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

    # Measure time
    inference_time = 0.0

    dataset_config: DictConfig = config.datasets[dataset]
    cluster_threshold: int = dataset_config["cluster_threshold"]
    logger.info(f"Dataset: {dataset}, Cluster Threshold: {cluster_threshold}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(config.device)

    # Capture the auxiliary action accuracy
    corr_actions, total_actions = 0.0, 0.0
    oracle_evaluator, evaluator = CorefEvaluator(), CorefEvaluator()
    coref_predictions, subtoken_maps = {}, {}

    logger.info(f"Evaluating on {len(data_iter_map[split][dataset])} examples")

    log_dir, log_file, log_link_file = get_log_file_name(
        config,
        dataset,
        teacher_force,
        gold_mentions,
        split,
        _iter,
    )

    f = open(log_file, "w")
    f_link = open(log_link_file, "w")

    for example in data_iter_map[split][dataset]:
        start_time = time.time()
        (
            pred_mentions,
            pred_mentions_emb,
            rep_emb_list,
            mention_scores,
            gt_actions,
            pred_actions,
            coref_scores,
            entity_cluster_states,
            link_time,
        ) = model(example, teacher_force=teacher_force, gold_mentions=gold_mentions)

        # Process predicted clusters
        raw_predicted_clusters = action_sequences_to_clusters(
            pred_actions, pred_mentions
        )
        predicted_clusters = filter_clusters(
            raw_predicted_clusters, threshold=cluster_threshold
        )
        mention_to_predicted = get_mention_to_cluster(predicted_clusters)

        gold_clusters = filter_clusters(
            example["clusters"], threshold=cluster_threshold
        )
        mention_to_gold = get_mention_to_cluster(gold_clusters)
        evaluator.update(
            predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold
        )

        elapsed_time = time.time() - start_time
        inference_time += elapsed_time

        coref_predictions[example["doc_key"]] = predicted_clusters
        if "orig_subtoken_map" in example:
            subtoken_maps[example["doc_key"]] = example["orig_subtoken_map"]
        else:
            subtoken_maps[example["doc_key"]] = example["subtoken_map"]

        total_actions += len(pred_actions)

        # Oracle clustering - Best performance possible given the predicted mentions
        oracle_clusters = action_sequences_to_clusters(gt_actions, pred_mentions)
        oracle_clusters = filter_clusters(oracle_clusters, threshold=cluster_threshold)
        mention_to_oracle = get_mention_to_cluster(oracle_clusters)
        oracle_evaluator.update(
            oracle_clusters, gold_clusters, mention_to_oracle, mention_to_gold
        )

        log_example = dict(example)
        log_example["pred_mentions"] = pred_mentions

        ## Removing from logs atm, should not be used anywhere
        # log_example["pred_mentions_emb"] = pred_mentions_emb

        log_example["rep_emb_list"] = rep_emb_list
        # print(pred_mentions_emb)
        log_example["mention_scores"] = mention_scores
        if cluster_threshold != 1:
            # For cluster threshold 1, raw and processed clusters are one and the same
            log_example["raw_predicted_clusters"] = raw_predicted_clusters

        log_example["gt_actions"] = gt_actions
        log_example["pred_actions"] = pred_actions
        log_example["predicted_clusters"] = predicted_clusters
        log_example["coref_scores"] = coref_scores

        if entity_cluster_states is not None:
            for key_entity in entity_cluster_states:
                log_example[key_entity] = entity_cluster_states[key_entity]

        del log_example["tensorized_sent"]
        for key in list(log_example.keys()):
            if isinstance(log_example[key], Tensor):
                del log_example[key]

        log_link_example = {
            "doc_key": example["doc_key"],
            "num_mentions": len(pred_mentions),
            "link_time": link_time,
        }

        ## Write to log files in the final test evaluation or only inference time
        if _iter == "":
            f.write(json.dumps(log_example) + "\n")
            f_link.write(json.dumps(log_link_example) + "\n")

    ## Close file handlers
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
    logger.info("F-score: %.1f %s" % (result_dict["fscore"], perf_str))

    logger.info("Oracle F-score: %.3f" % oracle_evaluator.get_prf()[2])
    logger.info(path.abspath(log_file))
    logger.handlers[0].flush()

    logger.info("Inference time: %.2f" % inference_time)
    max_mem = (
        (torch.cuda.max_memory_allocated(config.device) / (1024**3))
        if torch.cuda.is_available()
        else 0.0
    )
    logger.info("Max inference memory: %.1f GB" % max_mem)

    return result_dict


def targeted_coref_evaluation(
    config: DictConfig,
    model: EntityRankingModel,
    data_iter_map: Dict,
    dataset: str,
    split="test",
) -> Dict:
    raise NotImplementedError("Targeted evaluation not implemented yet.")


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

    dataset_config = config.datasets[dataset]
    if dataset_config.get("targeted_eval", False):
        return targeted_coref_evaluation(
            config, model, data_iter_map, dataset, split=split
        )
    else:
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
