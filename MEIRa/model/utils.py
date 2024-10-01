from coref_utils.utils import get_mention_to_cluster_idx
from collections import defaultdict


def get_gt_actions(pred_mentions, document, mem_type_config, mapped_mentions=[]):
    if "clusters" in document:
        # Ground truth is avaliable
        gt_clusters = document["clusters"]
        return get_actions_unbounded_fast(pred_mentions, gt_clusters, mapped_mentions)
    else:
        # Don't have ground truth clusters i.e. running it in the wild
        # Generate dummy actions
        return [(-1, "i")] * len(pred_mentions)


def action_sequences_to_clusters(actions, mentions, num_major_entities):

    cell_to_clusters = defaultdict(list)
    for mention, (cell_idx, action_type) in zip(mentions, actions):
        if action_type == "i":
            continue
        elif action_type == "o":
            cell_to_clusters[num_major_entities].append(mention)
        else:
            cell_to_clusters[cell_idx].append(mention)

    clusters = [[] for _ in range(num_major_entities + 1)]
    for cell_idx, cluster in cell_to_clusters.items():
        clusters[cell_idx] = cluster

    return clusters


def get_cluster_to_cell(mapped_mentions, mention_to_cluster):
    cluster_to_cell = {}
    cell_counter = 0
    for mention in mapped_mentions:
        if tuple(mention) not in mention_to_cluster:
            print("Error: Mention not in mentions", tuple(mention))
        else:
            mention_cluster = mention_to_cluster[tuple(mention)]
            if mention_cluster not in cluster_to_cell:
                cluster_to_cell[mention_cluster] = cell_counter
                cell_counter += 1
    return cluster_to_cell


def get_actions_unbounded_fast(pred_mentions, gt_clusters, mapped_mentions=[]):
    actions = []
    num_clusters = len(gt_clusters)
    mention_to_cluster = get_mention_to_cluster_idx(gt_clusters)
    for idx, mention in enumerate(pred_mentions):
        if tuple(mention) not in mention_to_cluster:
            actions.append((num_clusters - 1, "o"))
        else:
            mention_cluster = mention_to_cluster[tuple(mention)]
            if mention_cluster == num_clusters - 1:
                actions.append((mention_cluster, "o"))
            else:
                actions.append((mention_cluster, "c"))
    return actions
