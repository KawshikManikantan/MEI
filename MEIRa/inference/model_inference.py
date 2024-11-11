import torch
from os import path
from model.utils import action_sequences_to_clusters
from model.entity_ranking_model import EntityRankingModel
from inference.tokenize_doc import tokenize_and_segment_doc, basic_tokenize_doc
from omegaconf import OmegaConf, open_dict
from transformers import AutoModel, AutoTokenizer
import spacy
import json
import pytorch_utils.utils as utils


class Inference:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.best_model_path = path.join(model_path, "best/model.pth")
        self._load_model()

        self.max_segment_len = self.config.model.doc_encoder.transformer.max_segment_len
        self.tokenizer = self.model.mention_proposer.doc_encoder.tokenizer

    def find_repr_and_clean(self, basic_tokenized_doc):
        ## Find marked representatives
        num_brackets = 0
        start_tok = 0

        tokens_new = []  ## Contains {{ and }}
        tokens_proc = []  ## Does not contain {{ and }}
        basic_tokenized_doc_proc = []  ## Does not contain {{ and }}
        skip_next = 0
        for sentence in basic_tokenized_doc:
            tokens_sent = []
            for token_ind, token in enumerate(sentence):
                if skip_next:
                    skip_next = 0
                    continue
                if token_ind + 1 < len(sentence):
                    if token == "{" and sentence[token_ind + 1] == "{":
                        tokens_new.append("{{")
                        skip_next = 1
                    elif token == "}" and sentence[token_ind + 1] == "}":
                        tokens_new.append("}}")
                        skip_next = 1
                    else:
                        tokens_new.append(token)
                        tokens_sent.append(token)
                else:
                    tokens_new.append(token)
                    tokens_sent.append(token)
            basic_tokenized_doc_proc.append(tokens_sent)
            tokens_proc.extend(tokens_sent)

        active_ent_toks = []
        ent_toks = []
        for word_ind, word in enumerate(tokens_new):
            if word == "{{":
                num_brackets += 1
                start_tok += 1
            elif word == "}}":
                num_brackets += 1
                active_ent_toks[-1].append(
                    word_ind - num_brackets
                )  ## Since we included the current bracket upfront
                new_entity = active_ent_toks.pop()
                ent_toks.append(new_entity)
            else:
                while start_tok > 0:
                    active_ent_toks.append([word_ind - num_brackets])
                    start_tok -= 1

        ent_names = []
        for ent in ent_toks:
            ent_names.append(" ".join(tokens_proc[ent[0] : ent[1] + 1]))

        print("Entities: ", ent_toks)
        print("Entity Names: ", ent_names)

        return basic_tokenized_doc_proc, ent_toks, ent_names

    def get_ts_from_st(self, subtoken_map, representatives):
        ts_map = {}
        for subtoken_ind, token_ind in enumerate(subtoken_map):
            if token_ind not in ts_map:
                ts_map[token_ind] = [subtoken_ind]
                if subtoken_ind != 0:
                    ts_map[token_ind - 1].append(subtoken_ind - 1)
        ent_toks_st = []
        for entity in representatives:
            start_st = ts_map[entity[0]][0]
            end_st = ts_map[entity[1]][-1]
            ent_toks_st.append((start_st, end_st))
        return ent_toks_st, ts_map

    def process_doc_str(self, document):
        # Raw document string. First perform basic tokenization before further tokenization.
        basic_tokenizer = spacy.load("en_core_web_trf")
        basic_tokenized_doc = basic_tokenize_doc(document, basic_tokenizer)
        basic_tokenized_doc, representatives, representatives_names = (
            self.find_repr_and_clean(basic_tokenized_doc)
        )
        tokenized_doc = tokenize_and_segment_doc(
            basic_tokenized_doc,
            self.tokenizer,
            max_segment_len=self.max_segment_len,
        )
        representatives, representatives_names = zip(
            *sorted(zip(representatives, representatives_names))
        )

        print("Representatives: ", representatives)
        print("Representative Names: ", representatives_names)

        ent_toks_st, ts_map = self.get_ts_from_st(
            tokenized_doc["subtoken_map"], representatives
        )
        return (
            basic_tokenized_doc,
            tokenized_doc,
            representatives,
            representatives_names,
            ent_toks_st,
            ts_map,
        )

    def _load_model(self):
        checkpoint = torch.load(self.best_model_path, map_location="cpu")
        self.config = checkpoint["config"]
        self.train_info = checkpoint["train_info"]

        if self.config.model.doc_encoder.finetune:
            # Load the document encoder params if encoder is finetuned
            doc_encoder_dir = path.join(
                path.dirname(self.best_model_path),
                self.config.paths.doc_encoder_dirname,
            )
            if path.exists(doc_encoder_dir):
                self.config.model.doc_encoder.transformer.model_str = doc_encoder_dir

        self.config.model.memory.thresh = 0.5
        self.model = EntityRankingModel(self.config.model, self.config.trainer)

        # Document encoder parameters will be loaded via the huggingface initialization
        self.model.load_state_dict(checkpoint["model"], strict=False)

        if torch.cuda.is_available():
            self.model.cuda(device=self.config.device)

        self.model.eval()

    @torch.no_grad()
    def perform_coreference(self, document, doc_name):
        if isinstance(document, str):
            (
                basic_tokenized_doc,
                tokenized_doc,
                ent_toks,
                ent_names,
                ent_toks_st,
                ts_map,
            ) = self.process_doc_str(document)
            tokenized_doc["representatives"] = ent_toks_st
            tokenized_doc["doc_key"] = doc_name
            tokenized_doc["clusters"] = []
        else:
            raise ValueError

        (
            pred_mentions,
            pred_mention_emb_list,
            mention_scores,
            gt_actions,
            pred_actions,
            coref_scores_doc,
            entity_cluster_states,
            link_time,
        ) = self.model(tokenized_doc)

        idx_clusters = action_sequences_to_clusters(
            pred_actions, pred_mentions, len(ent_toks_st)
        )

        subtoken_map = tokenized_doc["subtoken_map"]
        orig_tokens = tokenized_doc["orig_tokens"]

        clusters = []
        for idx_cluster in idx_clusters:
            cur_cluster = []
            for ment_start, ment_end in idx_cluster:
                cur_cluster.append(
                    (
                        (subtoken_map[ment_start], subtoken_map[ment_end]),
                        " ".join(
                            orig_tokens[
                                subtoken_map[ment_start] : subtoken_map[ment_end] + 1
                            ]
                        ),
                    )
                )

            clusters.append(cur_cluster)

        keys_tokenized_doc = list(tokenized_doc.keys())
        for key in keys_tokenized_doc:
            if type(tokenized_doc[key]) == torch.Tensor:
                del tokenized_doc[key]

        tokenized_doc["tensorized_sent"] = [
            sent.tolist() for sent in tokenized_doc["tensorized_sent"]
        ]

        return {
            "tokenized_doc": tokenized_doc["orig_tokens"],
            "clusters": clusters,
            # "subtoken_idx_clusters": idx_clusters,
            # "actions": pred_actions,
            # "mentions": pred_mentions,
            # "representative_embs": entity_cluster_states["mem"],
            "representative_names": ent_names,
        }


if __name__ == "__main__":
    ## Arg Parser
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Specify model path")
    parser.add_argument("-d", "--doc", type=str, help="Specify document path")
    parser.add_argument(
        "-g", "--gpu", type=str, default="cuda:0", help="Specify GPU device"
    )
    parser.add_argument(
        "--doc_name", type=str, default="eval_doc", help="Specify encoder name"
    )
    parser.add_argument("-r", "--results", type=str, help="Specify results path")

    args = parser.parse_args()

    model_str = args.model
    doc_str = args.doc
    model = Inference(model_str)

    doc_str = open(doc_str).read()

    output_dict = model.perform_coreference(doc_str, args.doc_name)

    print("Keys: ", output_dict.keys())
    # for cluster_ind, cluster in enumerate(output_dict["clusters"]):
    #     print(f"{cluster_ind}:", cluster)

    with open(args.results, "w") as f:
        json.dump(output_dict, f)
