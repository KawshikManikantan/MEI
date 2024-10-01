import numpy as np
from utils.metrics import cosine


def cosine_map(orig_results, major_entities, threshold):
    modified_results = {}
    for document in orig_results:
        golden_clusters = sorted(
            [sorted(cluster) for cluster in orig_results[document]["clusters"]]
        )  ## Sorting because major entities are defined in this order
        predicted_clusters = orig_results[document][
            "predicted_clusters"
        ]  ## Not sorted because the representations exist in this fashion
        entity_rep = np.array(orig_results[document]["rep_emb_list"])
        mem = np.array(orig_results[document]["mem"])

        if (
            threshold > 1
        ):  ## Will require some changes if this code doesn't produce the same output.
            filt_mask = np.array(
                [
                    0 if counter < threshold else 1
                    for counter in orig_results[document]["ent_counter"]
                ]
            )
            mem = mem[filt_mask == 1]

        matching = cosine(entity_rep, mem, True)
        mem_index = matching[:, 0].tolist()
        entity_index = matching[:, 1].tolist()

        mapped_clusters = []
        for i in range(entity_rep.shape[0]):
            if i in entity_index:
                mapped_clusters.append(mem_index[entity_index.index(i)])
            else:
                mapped_clusters.append(-1)

        modified_result_dict = {
            "doc_key": document,
            "golden_clusters": [],
            "predicted_clusters": [],
        }

        for entity_ind, major_entity in enumerate(
            major_entities[document]["entity_id"]
        ):
            modified_result_dict["golden_clusters"].append(
                golden_clusters[major_entity]
            )
            if mapped_clusters[entity_ind] != -1:
                modified_result_dict["predicted_clusters"].append(
                    predicted_clusters[mapped_clusters[entity_ind]]
                )
            else:
                modified_result_dict["predicted_clusters"].append([])

        other_list = []
        for cluster_ind, cluster in enumerate(golden_clusters):
            if cluster_ind not in major_entities[document]["entity_id"]:
                other_list.extend(cluster)
        modified_result_dict["golden_clusters"].append(other_list)

        other_list = []
        for cluster_ind, cluster in enumerate(predicted_clusters):
            if cluster_ind not in mapped_clusters:
                other_list.extend(cluster)

        modified_result_dict["predicted_clusters"].append(other_list)

        modified_results[document] = modified_result_dict
    return modified_results
