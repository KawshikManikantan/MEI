from utils.metrics import fuzzy


def fuzzy_map(orig_results, major_entities, docs_processed):
    modified_results = {}
    for document in orig_results:
        golden_clusters = sorted(
            [sorted(cluster) for cluster in orig_results[document]["clusters"]]
        )
        predicted_clusters = sorted(
            [
                sorted(cluster)
                for cluster in orig_results[document]["predicted_clusters"]
            ]
        )
        entity_rep_ment_ind = major_entities[document]["mention_inds"]

        mentions_vs_tbound = docs_processed[document]["mentions_vs_tbound"]
        st_vs_tbound = docs_processed[document]["subtoken_vs_token"]
        t_vs_str = docs_processed[document]["token_vs_tokenstr"]
        entity_rep_tbound = [
            mentions_vs_tbound[mention_ind] for mention_ind in entity_rep_ment_ind
        ]
        predicted_clusters_tbound = [
            [
                [st_vs_tbound[mention[0]], st_vs_tbound[mention[1]]]
                for mention in cluster
            ]
            for cluster in predicted_clusters
        ]

        entity_rep_str = [
            " ".join(t_vs_str[tbound[0] : tbound[1] + 1])
            for tbound in entity_rep_tbound
        ]
        predicted_clusters_str = [
            [" ".join(t_vs_str[tbound[0] : tbound[1] + 1]) for tbound in mention]
            for mention in predicted_clusters_tbound
        ]
        matching = fuzzy(predicted_clusters_str, entity_rep_str, return_matching=True)

        ## For each entity in major entities, find the index of the cluster it belongs to in predicted clusters
        entity_rep_index = matching[:, 0].tolist()
        mapped_cluster_index = matching[:, 1].tolist()
        mapped_clusters = []
        for entity in range(len(major_entities[document]["entity_id"])):
            if entity in entity_rep_index:
                entity_index = entity_rep_index.index(entity)
                mapped_clusters.append(mapped_cluster_index[entity_index])
            else:
                entity_index = -1
                mapped_clusters.append(entity_index)

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
