import os
import hydra
import copy

from utils.utils import read_jsonl, write_jsonl


def modify_gt(orig_split, major_entities_split):
    modified_split = copy.deepcopy(orig_split)
    for document in modified_split.keys():
        cluster_inds = major_entities_split[document]["cluster_inds"]
        mention_inds = major_entities_split[document]["mention_inds"]
        representative_embs = major_entities_split[document].get(
            "representative_embs", None
        )

        if len(cluster_inds):
            if representative_embs is None:
                mention_inds, cluster_inds = list(
                    zip(*sorted(zip(mention_inds, cluster_inds)))
                )
            else:
                print("There are representative embeddings")
                mention_inds, cluster_inds, representative_embs = list(
                    zip(*sorted(zip(mention_inds, cluster_inds, representative_embs)))
                )

        print(mention_inds, cluster_inds)
        clusters_mod = []
        others = []
        clusters_init = sorted(
            sorted(cluster) for cluster in orig_split[document]["clusters"]
        )

        for cluster_ind in cluster_inds:
            clusters_mod.append(clusters_init[cluster_ind])

        for cluster_ind, cluster in enumerate(clusters_init):
            if cluster_ind not in cluster_inds:
                others.extend(cluster)
        clusters_mod.append(sorted(others))

        modified_split[document]["clusters"] = clusters_mod
        modified_split[document]["num_clusters"] = len(clusters_mod)

        representative_mentions = mention_inds
        mentions = sorted([mention for cluster in clusters_mod for mention in cluster])
        mention_bounds = [
            mentions[mention_ind] for mention_ind in representative_mentions
        ]

        modified_split[document]["representatives"] = mention_bounds
        modified_split[document]["representative_embs"] = representative_embs
    return modified_split


def process_split(split_file, major_entity_files_split, split_file_mei):
    if split_file:
        dataset = read_jsonl(split_file)
        major_entities_split = read_jsonl(major_entity_files_split)
        dataset_mod = modify_gt(dataset, major_entities_split)
        for document in dataset_mod.keys():
            dataset[document]["representatives"] = dataset_mod[document][
                "representatives"
            ]
        write_jsonl(split_file_mei, dataset_mod)
        write_jsonl(split_file, dataset)


@hydra.main(config_path=f"{os.getcwd()}/configs/args", config_name="args")
def main(args):
    for dataset in args["datasets"]:
        train_file = args["datasets"][dataset].get("train_file", None)
        dev_file = args["datasets"][dataset].get("dev_file", None)
        test_file = args["datasets"][dataset].get("test_file", None)

        train_file_mei = args["datasets"][dataset].get("train_file_mei", None)
        dev_file_mei = args["datasets"][dataset].get("dev_file_mei", None)
        test_file_mei = args["datasets"][dataset].get("test_file_mei", None)

        major_entity_files_train_dataset = args["datasets"][dataset].get(
            "train_me", None
        )
        major_entity_files_dev_dataset = args["datasets"][dataset].get("dev_me", None)
        major_entity_files_test_dataset = args["datasets"][dataset].get("test_me", None)

        process_split(train_file, major_entity_files_train_dataset, train_file_mei)
        process_split(dev_file, major_entity_files_dev_dataset, dev_file_mei)
        process_split(test_file, major_entity_files_test_dataset, test_file_mei)


if __name__ == "__main__":
    main()
