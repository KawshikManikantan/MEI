import os
import argparse
import hydra

from utils.utils import read_jsonl, process_and_save_init_results
from evaluate.eval_func import evaluate


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

    print("Run Name:", run)

    dataset_configs = experiment_config["datasets"]
    split = experiment_config["split"]
    setup = experiment_config["setup"]

    ext_mentions = setup == "em"
    gold = setup == "gm"
    gold_str = "_gold(eval)" if gold else ""
    ext_mentions_str = "_ext_ment" if ext_mentions else ""

    for dataset in dataset_configs:
        print(dataset)
        ## Read Model Results
        sota_jsonl = (
            f"{model_base_path}/{dataset}/{split}{gold_str}{ext_mentions_str}.log.jsonl"
        )
        if not os.path.exists(sota_jsonl):
            print(f"File {sota_jsonl} does not exist")
            continue
        orig_results = read_jsonl(sota_jsonl)

        ## Derive the result destination from the experiment config
        result_destination = experiment_config["paths"]["result_destination"]
        result_file_destination = os.path.join(
            result_destination, dataset, split, "result.jsonl"
        )
        print("Result Destination:", result_file_destination)
        process_and_save_init_results(orig_results, result_file_destination)

    ## Evaluate the results
    evaluate(experiment_config)


if __name__ == "__main__":
    main()
