import torch
import torch.nn as nn

from model.mention_proposal import MentionProposalModule
from model.utils import get_gt_actions
from model.memory.entity_memory import EntityMemory
from torch.profiler import profile, record_function, ProfilerActivity

from typing import Dict, List, Tuple
from omegaconf import DictConfig
from torch import Tensor
from transformers import PreTrainedTokenizerFast

import logging
import random
from collections import defaultdict
import copy

import time

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger()


class EntityRankingModel(nn.Module):
    """
    Coreference model based on Entity-Ranking paradigm.

    In the entity-ranking paradigm, given a new mention we rank the different
    entity clusters to determine the clustering updates. Entity-Ranking paradigm
    allows for a naturally scalable solution to coreference resolution.
    Reference: Rahman and Ng [https://arxiv.org/pdf/1405.5202.pdf]

    This particular implementation represents the entities/clusters via fixed-dimensional
    dense representations, typically a simple avereage of mention representations.
    Clustering is performed in an online, autoregressive manner where mentions are
    processed in a left-to-right manner.
    References:
            Toshniwal et al [https://arxiv.org/pdf/2010.02807.pdf]
      Toshniwal et al [https://arxiv.org/pdf/2109.09667.pdf]
    """

    def __init__(self, model_config: DictConfig, train_config: DictConfig):
        super(EntityRankingModel, self).__init__()
        self.config = model_config
        self.train_config = train_config

        # Dropout module - Used during training
        self.drop_module = nn.Dropout(p=train_config.dropout_rate)

        self.loss_template_dict = {
            "total": torch.tensor(0.0, requires_grad=True),
            "ment_loss": torch.tensor(0.0),
            "coref": torch.tensor(0.0),
            "mention_count": torch.tensor(0.0),
            "ment_correct": torch.tensor(0.001),
            "ment_total": torch.tensor(0.001),
            "ment_tp": torch.tensor(0.001),
            "ment_pp": torch.tensor(0.001),
            "ment_ap": torch.tensor(0.001),
        }

        # Document encoder + Mention proposer
        self.mention_proposer = MentionProposalModule(
            self.config, train_config, drop_module=self.drop_module
        )

        # Clustering module
        span_emb_size: int = self.mention_proposer.span_emb_size
        # Use of genre feature in clustering or not
        if self.config.metadata_params.use_genre_feature:
            self.config.memory.num_feats = 3

        self.mem_type = self.config.memory.mem_type.name

        self.memory_net = EntityMemory(
            config=self.config.memory,
            span_emb_size=span_emb_size,
            drop_module=self.drop_module,
        )

        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=self.train_config.label_smoothing_wt
        )

        if self.config.metadata_params.use_genre_feature:
            self.genre_embeddings = nn.Embedding(
                num_embeddings=len(self.config.metadata_params.genres),
                embedding_dim=self.config.mention_params.emb_size,
            )

    @property
    def device(self) -> torch.device:
        return self.mention_proposer.device

    def get_params(self, named=False) -> Tuple[List, List]:
        """Returns a tuple of document encoder parameters and rest of the model params."""

        encoder_params, mem_params = [], []
        for name, param in self.named_parameters():
            elem = (name, param) if named else param
            if "doc_encoder" in name:
                encoder_params.append(elem)
            else:
                mem_params.append(elem)

        return encoder_params, mem_params

    def get_tokenizer(self) -> PreTrainedTokenizerFast:
        """Returns tokenizer used by the document encoder."""

        return self.mention_proposer.doc_encoder.get_tokenizer()

    def get_metadata(self, document: Dict) -> Dict:
        """Extract metadata such as document genre from document."""

        meta_params = self.config.metadata_params
        if meta_params.use_genre_feature:
            doc_class = document["doc_key"][:2]
            if doc_class in meta_params.genres:
                doc_class_idx = meta_params.genres.index(doc_class)
            else:
                doc_class_idx = meta_params.genres.index(
                    meta_params.default_genre
                )  # Default genre

            return {
                "genre": self.genre_embeddings(
                    torch.tensor(doc_class_idx, device=self.device)
                )
            }
        else:
            return {}

    def calculate_coref_loss(
        self, action_prob_list: List, action_tuple_list: List[Tuple[int, str]]
    ) -> Tensor:
        """Calculates the coreference loss for the autoregressive online clustering module.

        Args:
                action_prob_list (List):
                        Probability of each clustering action i.e. mention is merged with existing clusters
                        or a new cluster is created.
                action_tuple_list (List[Tuple[int, str]]):
                        Ground truth actions represented as a tuple of cluster index and action string.
                        'c' represents that the mention is coreferent with existing clusters while
                        'o' represents that the mention represents a new cluster.

        Returns:
                coref_loss (torch.Tensor):
                        The scalar tensor representing the coreference loss.
        """
        counter = 0
        correct = 0
        coref_loss = torch.tensor(0.0, device=self.device)
        num_predictions_clusters = defaultdict(int)
        for idx, (cell_idx, action_str) in enumerate(action_tuple_list):
            if action_str == "c":
                ## Major Entity
                gt_idx = cell_idx

            elif action_str == "o":
                ## Other Entity
                gt_idx = action_prob_list[counter].shape[0] - 1

            else:
                continue

            target = torch.tensor([gt_idx], device=self.device)
            if target[0] == torch.argmax(
                torch.unsqueeze(action_prob_list[counter], dim=0)
            ):
                correct += 1
            num_predictions_clusters[
                torch.argmax(torch.unsqueeze(action_prob_list[counter], dim=0)).item()
            ] += 1
            coref_loss += self.loss_fn(
                torch.unsqueeze(action_prob_list[counter], dim=0), target
            )

            counter += 1
        return coref_loss

    @staticmethod
    def get_filtered_clusters(
        clusters,
        init_token_offset,
        final_token_offset,
        cluster_mask=None,
        with_offset=True,
    ):
        """Filter clusters from a document given the token offsets."""
        """Note that len(cluster_mask) == len(clusters) assured in the previous function."""

        filt_clusters = []
        no_rep_cluster_mentions = (
            []
        )  ## Mentions that belonged to a major entity whose representative phrase is not part of the current mentions.
        for cluster_ind, orig_cluster in enumerate(clusters):
            cluster = []
            for ment_start, ment_end in orig_cluster:
                if ment_start >= init_token_offset and ment_end < final_token_offset:
                    if with_offset:
                        cluster.append((ment_start, ment_end))
                    else:
                        cluster.append(
                            (
                                ment_start - init_token_offset,
                                ment_end - init_token_offset,
                            )
                        )
            if len(cluster) != 0:
                if (
                    cluster_mask
                ):  ## During this process if we missed any representative phrases, all clusters that have no representative phrase will be added to the last cluster.
                    if (
                        cluster_mask[cluster_ind] == True
                    ):  ## If representative phrase is in the current segment then, there exists atleast one mention that belongs to the cluster. But anyways
                        filt_clusters.append(cluster)
                    else:
                        no_rep_cluster_mentions.extend(cluster)
                else:
                    filt_clusters.append(cluster)

        if cluster_mask:
            if len(filt_clusters) == 0:
                filt_clusters.append(no_rep_cluster_mentions)
            else:
                filt_clusters[-1].extend(no_rep_cluster_mentions)

        return filt_clusters

    @staticmethod
    def get_filtered_representatives(
        representatives, init_token_offset, final_token_offset, with_offset=True
    ):
        """Filter clusters from a document given the token offsets."""
        filt_reps = []
        indices = []
        for rep_ind, (ment_start, ment_end) in enumerate(representatives):
            if ment_start >= init_token_offset and ment_end < final_token_offset:
                if with_offset:
                    filt_reps.append((ment_start, ment_end))
                else:
                    filt_reps.append(
                        (
                            ment_start - init_token_offset,
                            ment_end - init_token_offset,
                        )
                    )
                indices.append(rep_ind)
        return filt_reps, indices

    @staticmethod
    def mask_representative_phrases(rep_emb_list):
        positive_inds = []
        for rep_emb_ind, rep_emb in enumerate(rep_emb_list):
            if not isinstance(rep_emb, int):
                positive_inds.append(rep_emb_ind)

        if len(positive_inds) > 1:
            num_entitites_preserved = random.randint(1, len(positive_inds))
            random.shuffle(positive_inds)
            for ind in positive_inds[num_entitites_preserved:]:
                rep_emb_list[ind] = -1

        return rep_emb_list

    def forward_training(self, document: Dict) -> Dict:
        """Forward pass for training.

        Args:
                document: The tensorized document.

        Returns:
                loss_dict (Dict): Loss dictionary containing the losses of different stages of the model.
        """
        # print(document["doc_key"])

        assert (
            len(document["clusters"]) == len(document["representatives"]) + 1
        ), "Length of clusters not equal to length of representatives + 1."
        assert document["representatives"] == sorted(
            document["representatives"]
        ), "Representatives are not sorted."

        loss_dict = copy.deepcopy(self.loss_template_dict)
        max_training_segments = self.train_config.get("max_training_segments", None)
        num_segments = len(document["sentences"])
        if max_training_segments is None:
            seg_range = [0, num_segments]
        else:
            if num_segments > max_training_segments:
                start_seg = random.randint(0, num_segments - max_training_segments)
                seg_range = [start_seg, start_seg + max_training_segments]
            else:
                seg_range = [0, num_segments]

        # Initialize lists to track all the mentions predicted across the chunks
        pred_mentions_list, mention_emb_list, rep_emb_list = (
            [],
            [],
            [-1 for _ in range(len(document["representatives"]))],
        )
        init_token_offset = sum(
            [len(document["sentences"][idx]) for idx in range(0, seg_range[0])]
        )
        token_offset = init_token_offset

        # Metadata such as document genre can be used by model for clustering
        metadata = self.get_metadata(document)

        # Initialize the mention loss
        ment_loss = None

        # Step 1: Predict all the mentions
        for idx in range(seg_range[0], seg_range[1]):

            num_tokens = len(document["sentences"][idx])

            representatives_entities, rep_filtered_inds = (
                self.get_filtered_representatives(
                    document["representatives"],
                    token_offset,
                    token_offset + num_tokens,
                    with_offset=False,
                )
            )
            cur_doc_slice = {
                "tensorized_sent": document["tensorized_sent"][idx],
                "sentence_map": document["sentence_map"][
                    token_offset : token_offset + num_tokens
                ],
                "subtoken_map": document["subtoken_map"][
                    token_offset : token_offset + num_tokens
                ],
                "sent_len_list": [document["sent_len_list"][idx]],
                "clusters": self.get_filtered_clusters(
                    document["clusters"],
                    token_offset,
                    token_offset + num_tokens,
                    with_offset=False,
                ),
                "representatives": representatives_entities,
                "doc_key": document["doc_key"],
            }

            ## No golden mentions in the current segment and mode is golden so basically no job to do.
            if (
                len(cur_doc_slice["clusters"]) == 0
                and self.mention_proposer.config.mention_params.use_gold_ments
            ):
                token_offset += num_tokens
                continue

            proposer_output_dict = self.mention_proposer(cur_doc_slice, eval_loss=True)

            ### Shifted above because if the model predicts no mentions then earlier it had no mention loss. But now it has.
            if "ment_loss" in proposer_output_dict:
                if ment_loss is None:
                    ment_loss = proposer_output_dict["ment_loss"]
                else:
                    ment_loss += proposer_output_dict["ment_loss"]

            ## If no mentions are predicted, originally then no coref loss and surprisingly no mention loss as well :)
            if proposer_output_dict.get("ments", None) is None:
                token_offset += num_tokens
                continue

            ## Mention post-processing and collection happens here: Add the document offset to mentions predicted for the current chunk
            cur_pred_mentions = proposer_output_dict.get("ments") + token_offset
            pred_mentions_list.extend(cur_pred_mentions.tolist())
            mention_emb_list.extend(proposer_output_dict["ment_emb_list"])
            for key in ["ment_correct", "ment_total", "ment_tp", "ment_pp", "ment_ap"]:
                if key in proposer_output_dict:
                    loss_dict[key] += proposer_output_dict[key]

            ## Collect representation embeddings:
            for ind, rep_ind in enumerate(rep_filtered_inds):
                rep_emb_list[rep_ind] = proposer_output_dict["rep_emb_list"][ind]

            # Update the document offset for next iteration
            token_offset += num_tokens

        ## Collect mention detection loss
        if ment_loss is not None:
            ## Tried training the model with only mention loss, but it did not work well.
            if self.train_config.ment_loss_incl:
                loss_dict["total"] = ment_loss
            loss_dict["ment_loss"] = ment_loss

        # Step 2: Perform clustering
        # Get clusters part of the truncated document

        ## select certain entities or representatives
        if self.train_config.get("generalise", False):
            rep_emb_list = self.mask_representative_phrases(rep_emb_list)

        rep_emb_list_filtered = []
        entities_mask = []

        for rep_emb in rep_emb_list:
            if not isinstance(rep_emb, int):
                rep_emb_list_filtered.append(rep_emb)
                entities_mask.append(True)
            else:
                entities_mask.append(False)
        ## For the other cluster that contains all the mentions that do not belong to any representative phrase.
        entities_mask.append(True)

        truncated_document_clusters = {
            "clusters": self.get_filtered_clusters(
                document["clusters"],
                init_token_offset,
                token_offset,
                cluster_mask=entities_mask,
            )
        }

        assert (
            len(document["clusters"]) == len(document["representatives"]) + 1
        ), "Number of clusters and representatives after segmentation do not match."

        # Get ground truth clustering mentions
        gt_actions: List[Tuple[int, str]] = get_gt_actions(
            pred_mentions_list, truncated_document_clusters, self.config.memory.mem_type
        )

        pred_mentions = torch.tensor(pred_mentions_list, device=self.device)

        if (
            len(rep_emb_list_filtered) == 0
        ):  ## No representative phrases in the current segments, so no coref loss
            return loss_dict

        coref_new_list = self.memory_net.forward_training(
            pred_mentions, mention_emb_list, rep_emb_list_filtered, gt_actions, metadata
        )

        if len(coref_new_list) > 0:
            coref_loss = self.calculate_coref_loss(coref_new_list, gt_actions)
            loss_dict["total"] = loss_dict["total"] + coref_loss
            loss_dict["coref"] = coref_loss
            loss_dict["mention_count"] += torch.tensor(len(coref_new_list))

        return loss_dict

    def forward(self, document: Dict, teacher_force=False, gold_mentions=False):
        """Forward pass of the streaming coreference model.

        This method performs streaming coreference. The entity clusters from previous
        documents chunks are represented as vectors and passed along to the processing
        of subsequent chunks along with the metadata associated with these clusters.

        Args:
                document (Dict): Tensorized document

        Returns:
                 pred_mentions_list (List): Mentions predicted by the mention proposal module
                 mention_scores (List): Scores assigned by the mention proposal module for
                      the predicted mentions
                 gt_actions (List): Ground truth clustering actions; useful for calculating oracle performance
                 action_list (List): Actions predicted by the clustering module for the predicted mentions
        '"""

        # Initialize lists to track all the actions taken, mentions predicted across the chunks
        assert document["representatives"] == sorted(
            document["representatives"]
        ), "Representatives are not sorted."

        pred_mentions_list, pred_mention_emb_list, mention_scores, pred_actions = (
            [],
            [],
            [],
            [],
        )
        # Initialize entity clusters and current document token offset
        entity_cluster_states, token_offset = None, 0

        metadata = self.get_metadata(document)
        coref_scores_doc = []

        link_time = 0.0

        for idx in range(0, len(document["sentences"])):
            num_tokens = len(document["sentences"][idx])

            new_representatives_entities, rep_filtered_inds = (
                self.get_filtered_representatives(
                    document["representatives"],
                    token_offset,
                    token_offset + num_tokens,
                    with_offset=False,
                )
            )

            ext_predicted_mentions_filt, _ = self.get_filtered_representatives(
                document.get("ext_predicted_mentions", []),
                token_offset,
                token_offset + num_tokens,
                with_offset=False,
            )

            cur_example = {
                "tensorized_sent": document["tensorized_sent"][idx],
                "sentence_map": document["sentence_map"][
                    token_offset : token_offset + num_tokens
                ],
                "subtoken_map": document["subtoken_map"][
                    token_offset : token_offset + num_tokens
                ],
                "sent_len_list": [document["sent_len_list"][idx]],
                "clusters": self.get_filtered_clusters(
                    document["clusters"],
                    token_offset,
                    token_offset + num_tokens,
                    with_offset=False,
                ),
                "representatives": new_representatives_entities,
                "ext_predicted_mentions": ext_predicted_mentions_filt,
            }

            # Pass along other metadata
            for key in document:
                if key not in cur_example:
                    cur_example[key] = document[key]

            if len(cur_example["clusters"]) == 0 and (
                self.mention_proposer.config.mention_params.use_gold_ments
                or gold_mentions
            ):
                token_offset += num_tokens
                continue

            proposer_output_dict = self.mention_proposer(
                cur_example, gold_mentions=gold_mentions
            )

            if proposer_output_dict.get("ments", None) is None:
                token_offset += num_tokens
                continue

            # Add the document offset to mentions predicted for the current chunk
            # It's important to add the offset before clustering because features like
            # number of tokens between the last mention of the cluster and the current mention
            # will be affected if the current token indices of the mention are not supplied.
            cur_pred_mentions = proposer_output_dict.get("ments") + token_offset

            # Update the document offset for next iteration
            token_offset += num_tokens

            # Get ground truth clustering mentions
            pred_mentions_list.extend(cur_pred_mentions.tolist())
            gt_actions_full: List[Tuple[int, str]] = get_gt_actions(
                pred_mentions_list, document, self.config.memory.mem_type
            )
            gt_actions = gt_actions_full[-len(cur_pred_mentions.tolist()) :]

            pred_mention_emb_list.extend(
                [emb.tolist() for emb in proposer_output_dict.get("ment_emb_list")]
            )
            mention_scores.extend(proposer_output_dict["ment_scores"].tolist())

            start_time = time.time()

            repr_candidates = list(proposer_output_dict["rep_emb_list"])

            # Pass along entity clusters from previous chunks while processing next chunks
            cur_pred_actions, entity_cluster_states, coref_scores_list = (
                self.memory_net(
                    cur_pred_mentions,
                    list(proposer_output_dict["ment_emb_list"]),
                    repr_candidates,
                    gt_actions,
                    metadata,
                    teacher_force=teacher_force,
                    memory_init=entity_cluster_states,
                )
            )
            link_time += time.time() - start_time
            # print(
            #     "Number of representatives available now: ",
            #     entity_cluster_states["mem"].shape[0],
            # )
            pred_actions.extend(cur_pred_actions)
            coref_scores_doc.extend(coref_scores_list)

        gt_actions = get_gt_actions(
            pred_mentions_list, document, self.config.memory.mem_type
        )  # Useful for oracle calcs

        for ind in range(len(coref_scores_doc)):
            coref_scores_doc[ind] = coref_scores_doc[ind].tolist()

        if entity_cluster_states is not None:
            for key in entity_cluster_states:
                entity_cluster_states[key] = entity_cluster_states[key].tolist()

        return (
            pred_mentions_list,
            pred_mention_emb_list,
            mention_scores,
            gt_actions,
            pred_actions,
            coref_scores_doc,
            entity_cluster_states,
            link_time,
        )
