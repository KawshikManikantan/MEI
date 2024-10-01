import json
from os import path
from typing import Dict
import jsonlines


def get_data_file(data_dir: str, split: str, max_segment_len: int) -> str:
    jsonl_file = path.join(data_dir, "{}.{}.jsonlines".format(split, max_segment_len))
    print("File access: ", jsonl_file)
    if path.exists(jsonl_file):
        return jsonl_file
    else:
        jsonl_file = path.join(data_dir, "{}.jsonlines".format(split))
        if path.exists(jsonl_file):
            return jsonl_file


def load_dataset(
    data_dir: str,
    singleton_file: str = None,
    max_segment_len: int = 2048,
    num_train_docs: int = None,
    num_dev_docs: int = None,
    num_test_docs: int = None,
    dataset_name: str = None,
) -> Dict:
    all_splits = []
    for split in ["train", "dev", "test"]:
        jsonl_file = get_data_file(data_dir, split, max_segment_len)
        if jsonl_file is None:
            raise ValueError(f"No relevant files at {data_dir}")
        split_data = []
        with open(jsonl_file) as f:
            for line in f:
                load_dict = json.loads(line.strip())
                load_dict["dataset_name"] = dataset_name
                split_data.append(load_dict)
        all_splits.append(split_data)

    train_data, dev_data, test_data = all_splits

    # train_data[""]

    if singleton_file is not None and path.exists(singleton_file):
        num_singletons = 0
        with open(singleton_file) as f:
            singleton_data = json.loads(f.read())

        for instance in train_data:
            doc_key = instance["doc_key"]
            if doc_key in singleton_data:
                num_singletons += len(singleton_data[doc_key])
                instance["clusters"].extend(singleton_data[doc_key])

        print("Added %d singletons" % num_singletons)

    print("NUM TEST DOCS: ", num_test_docs)
    print("TEST DATA: ", len(test_data[:num_test_docs]))
    return {
        "train": train_data[:num_train_docs],
        "dev": dev_data[:num_dev_docs],
        "test": test_data[:num_test_docs],
    }


def load_eval_dataset(
    data_dir: str,
    external_md_file: str,
    max_segment_len: int,
    dataset_name: str = None,
    partial=False,
) -> Dict:
    data_dict = {}
    for split in ["dev", "test"]:
        jsonl_file = get_data_file(data_dir, split, max_segment_len)
        if jsonl_file is not None:
            split_data = []
            with open(jsonl_file) as f:
                for line in f:
                    load_dict = json.loads(line.strip())
                    load_dict["dataset_name"] = dataset_name
                    split_data.append(load_dict)

            data_dict[split] = split_data

    if external_md_file is not None and path.exists(external_md_file):
        predicted_mentions = {}
        with jsonlines.open(external_md_file, mode="r") as reader:
            for line in reader:
                predicted_mentions[line["doc_key"]] = line
        for split in ["dev", "test"]:
            for instance in data_dict[split]:
                doc_key = instance["doc_key"]
                if doc_key in predicted_mentions:
                    instance["ext_predicted_mentions"] = sorted(
                        predicted_mentions[doc_key]["pred_mentions"]
                    )

    return data_dict
