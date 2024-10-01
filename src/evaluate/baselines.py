import os
import hydra
import argparse
from omegaconf import OmegaConf

from utils.utils import (
    get_major_entities,
    read_jsonl,
    get_coref_docs,
    write_jsonl,
    process_and_save_init_results,
)

from utils.get_processed_dataset import get_processed_dataset
from evaluate.eval_func import evaluate

from evaluate.coref_maps.coref_fm import fuzzy_map
from evaluate.coref_maps.coref_cm import cosine_map


@hydra.main(config_path=f"{os.getcwd()}/configs/args", config_name="args")
def main(experiment_config):
    model_base_path = experiment_config["paths"]["model_base_path"]
    run = experiment_config["run"]
    if model_base_path is None or not os.path.exists(model_base_path):
        print("Model path base is not provided")
        return

    if run is None:
        ## Set run_name as the model name
        experiment_config["run"] = model_base_path.split("/")[-1]

    dataset_configs = experiment_config["datasets"]
    split = experiment_config["split"]
    setup = experiment_config["setup"]
    map_mode = experiment_config["mode"][:-1]  ## Removing the last character /

    ext_mentions = setup == "em"
    gold = setup == "gm"
    ext_mentions_str = "_ext_ment" if ext_mentions else ""
    gold_str = "_gold(eval)" if gold else ""

    for dataset in dataset_configs:
        doc_addr = dataset_configs[dataset][f"{split}_file"]
        doc_tsv_addr = dataset_configs[dataset]["tsv"]
        doc_me = dataset_configs[dataset][f"{split}_me"]
        doc_threshold = dataset_configs[dataset][f"cluster_threshold"]
        docs = get_coref_docs(doc_addr)
        docs_processed = get_processed_dataset(docs, doc_tsv_addr, head=False)
        major_entities = get_major_entities(doc_me)

        ## Read Model Results
        if map_mode == "coref_id":
            sota_jsonl = f"{model_base_path}/{dataset}/coref_id/{split}{gold_str}{ext_mentions_str}.log.jsonl"
        else:
            sota_jsonl = f"{model_base_path}/{dataset}/{split}{gold_str}{ext_mentions_str}.log.jsonl"

        ## Load required data:
        orig_results = read_jsonl(sota_jsonl)

        ## Derive the init esult destination from the experiment config
        result_destination = experiment_config["paths"]["result_destination"]
        init_result_file_destination = os.path.join(
            result_destination, dataset, split, "init_result.jsonl"
        )
        print("Init Result Destination:", init_result_file_destination)
        process_and_save_init_results(orig_results, init_result_file_destination)

        ## Derive the result destination from the experiment config
        result_file_destination = os.path.join(
            result_destination, dataset, split, "result.jsonl"
        )

        ## Get results based on the mode
        if map_mode == "coref_id":
            modified_results = orig_results
        elif map_mode == "coref_fm":
            modified_results = fuzzy_map(orig_results, major_entities, docs_processed)
        elif map_mode == "coref_cm":
            modified_results = cosine_map(orig_results, major_entities, doc_threshold)
        else:
            print("Invalid mode")
            return

        write_jsonl(result_file_destination, modified_results)

    ## Evaluate the results
    evaluate(experiment_config)


if __name__ == "__main__":
    main()
