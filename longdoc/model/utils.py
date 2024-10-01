from coref_utils.utils import get_mention_to_cluster_idx


def get_gt_actions(pred_mentions, document, mem_type_config, mapped_mentions=[]):
    if "clusters" in document:
        # Ground truth is avaliable
        gt_clusters = document["clusters"]
        if mem_type_config.name == "unbounded":
            return get_actions_unbounded_fast(
                pred_mentions, gt_clusters, mapped_mentions
            )
        else:
            raise NotImplementedError
    else:
        # Don't have ground truth clusters i.e. running it in the wild
        # Generate dummy actions
        return [(-1, "i")] * len(pred_mentions)


def action_sequences_to_clusters(actions, mentions):
    clusters = []
    cell_to_clusters = {}

    for mention, (cell_idx, action_type) in zip(mentions, actions):
        if action_type == "c":
            cell_to_clusters[cell_idx].append(mention)
        elif action_type == "o":
            # Overwrite
            if cell_idx in cell_to_clusters:
                # Remove the old cluster and initialize the new
                clusters.append(cell_to_clusters[cell_idx])
            cell_to_clusters[cell_idx] = [mention]
        elif action_type == "n":
            clusters.append([mention])

    for cell_idx, cluster in cell_to_clusters.items():
        clusters.append(cluster)

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
    mention_to_cluster = get_mention_to_cluster_idx(gt_clusters)
    cluster_to_cell = get_cluster_to_cell(mapped_mentions, mention_to_cluster)
    cell_counter = len(cluster_to_cell)
    # print(mention_to_cluster)
    for idx, mention in enumerate(pred_mentions):
        if tuple(mention) not in mention_to_cluster:
            actions.append((-1, "i"))
        else:
            mention_cluster = mention_to_cluster[tuple(mention)]
            if mention_cluster in cluster_to_cell:
                # Cluster is already being tracked
                actions.append((cluster_to_cell[mention_cluster], "c"))
            else:
                # Cluster is not being tracked
                # Add the mention to being tracked
                cluster_to_cell[mention_cluster] = cell_counter
                actions.append((cell_counter, "o"))
                cell_counter += 1

    return actions
