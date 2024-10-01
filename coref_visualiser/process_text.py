from flask import Markup

def get_tbound_info(clusters_st,doc_processed):
   
    clusters_t = []
    st_vs_tbound = doc_processed["subtoken_vs_token"]
    for cluster in clusters_st:
        cluster_t = []
        for mention in cluster:
            start = mention[0]
            end = mention[1]
            cluster_t.append([st_vs_tbound[start],st_vs_tbound[end]])
        clusters_t.append(cluster_t)
    return clusters_t

def gold_full(text,results_doc,doc_processed,major_entities):
    golden_clusters_full = results_doc["golden_clusters"][:-1]
    # golden_clusters_full = results_doc["clusters"]
    print(len(golden_clusters_full))
    golden_clusters_full_tbound = get_tbound_info(golden_clusters_full,doc_processed)
        
    # Create a list to store the formatted words
    formatted_words = text.split()
    
    all_mentions_pos = []
    for cluster_ind,cluster in enumerate(golden_clusters_full_tbound):
        for mention in cluster:
            mention = tuple(mention)
            all_mentions_pos.append((mention[0],"<span style='color:blue;'>["))
            all_mentions_pos.append((mention[1],f"]<sub>{cluster_ind}</sub></span>"))
    all_mentions_pos.sort()

    for pos,element in reversed(all_mentions_pos):
        if element[-1] == "[":
            formatted_words[pos] = element + formatted_words[pos]
        else:
            formatted_words[pos] = formatted_words[pos] + element

    return Markup(' '.join(formatted_words))

def gold_head(text,results_doc,doc_processed,major_entities):
    golden_clusters_head = results_doc["golden_clusters_head"]
    # Create a list to store the formatted words
    formatted_words = text.split()
    
    for cluster_ind,cluster in enumerate(golden_clusters_head):
        for mention in cluster:
            mention = tuple(mention)
            word_ind = mention[0]
            formatted_words[word_ind] = f"<span style='color:blue;'>{formatted_words[word_ind]}<sub>{cluster_ind}</sub></span>"

    return Markup(' '.join(formatted_words))

def predicted_head(text,results_doc,doc_processed,major_entities):
    golden_clusters_head = results_doc["golden_clusters_head"][:-1]
    predicted_clusters_head = results_doc["predicted_clusters_head"][:-1]
    formatted_words = text.split()

    for cluster_ind,cluster in enumerate(predicted_clusters_head):
        for mention in cluster:
            word_ind = mention[0]
            if mention in golden_clusters_head[cluster_ind]:
                formatted_words[word_ind] = f"<span style='color:blue;'>{formatted_words[word_ind]}<sub>{cluster_ind}</sub></span>"
            else:
                formatted_words[word_ind] = f"<span style='color:red;'>{formatted_words[word_ind]}<sub>{cluster_ind}</sub></span>"
    
    predicted_mentions_head = sorted([mention for cluster in predicted_clusters_head for mention in cluster])
    for cluster_ind,cluster in enumerate(golden_clusters_head):
        for mention in cluster:
            if mention not in predicted_mentions_head:
                word_ind = mention[0]
                formatted_words[word_ind] = f"<span style='color:red;'>{formatted_words[word_ind]}</sub></span>"

    return Markup(' '.join(formatted_words))

def predicted_full(text,results_doc,doc_processed,major_entities):
    golden_clusters_full = results_doc["golden_clusters"][:-1]
    # golden_clusters_full = results_doc["clusters"]
    # print(len(golden_clusters_full))
    golden_clusters_full_tbound = get_tbound_info(golden_clusters_full,doc_processed)
    predicted_clusters_full = results_doc["predicted_clusters"][:-1]
    # predicted_clusters_full = results_doc["predicted_clusters"]
    predicted_clusters_full_tbound = get_tbound_info(predicted_clusters_full,doc_processed)
    formatted_words = text.split()

    all_mentions_pos = []

    for cluster_ind,cluster in enumerate(predicted_clusters_full_tbound):
        for mention in cluster:
            word_ind = mention[0]
            if mention in golden_clusters_full_tbound[cluster_ind]:
                all_mentions_pos.append((mention[0],"<span style='color:blue;'>["))
                all_mentions_pos.append((mention[1],f"]<sub>{cluster_ind}</sub></span>"))
            else:
                all_mentions_pos.append((mention[0],"<span style='color:red;'>["))
                all_mentions_pos.append((mention[1],f"]<sub>{cluster_ind}</sub></span>"))
    
    predicted_mentions_full = sorted([mention for cluster in predicted_clusters_full_tbound for mention in cluster])
    for cluster_ind,cluster in enumerate(golden_clusters_full_tbound):
        for mention in cluster:
            if mention not in predicted_mentions_full:
                all_mentions_pos.append((mention[0],"<span style='color:red;'>["))
                all_mentions_pos.append((mention[1],f"]</span>"))
    
    all_mentions_pos.sort()

    for pos,element in reversed(all_mentions_pos):
        if element[-1] == "[":
            formatted_words[pos] = element + formatted_words[pos]
        else:
            formatted_words[pos] = formatted_words[pos] + element

    # print(' '.join(formatted_words))
    return Markup(' '.join(formatted_words))