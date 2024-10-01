import torch
from model.memory import BaseMemory
from pytorch_utils.modules import MLP
import torch.nn as nn

from omegaconf import DictConfig
from typing import Dict, Tuple, List
from torch import Tensor
from tqdm import tqdm
import math


class EntityMemory(BaseMemory):
    """Module for clustering proposed mention spans using Entity-Ranking paradigm."""

    def __init__(
        self, config: DictConfig, span_emb_size: int, drop_module: nn.Module
    ) -> None:
        super(EntityMemory, self).__init__(config, span_emb_size, drop_module)
        self.mem_type: DictConfig = config.mem_type

    def forward_training(
        self,
        ment_boundaries: Tensor,
        mention_emb_list: List[Tensor],
        rep_emb_list: List[Tensor],
        gt_actions: List[Tuple[int, str]],
        metadata: Dict,
    ) -> List[Tensor]:
        """
        Forward pass during coreference model training where we use teacher-forcing.

        Args:
                ment_boundaries: Mention boundaries of proposed mentions
                mention_emb_list: Embedding list of proposed mentions
                gt_actions: Ground truth clustering actions
                metadata: Metadata such as document genre

        Returns:
                coref_new_list: Logit scores for ground truth actions.
        """
        assert (
            len(rep_emb_list) != 0
        ), "There are no entity representations, should not happen."

        # Initialize memory
        coref_new_list = []

        mem_vectors, mem_vectors_init, ent_counter, last_mention_start = (
            self.initialize_memory(rep=rep_emb_list)
        )

        for ment_idx, (ment_emb, (gt_cell_idx, gt_action_str)) in enumerate(
            zip(mention_emb_list, gt_actions)
        ):

            ment_start, ment_end = ment_boundaries[ment_idx]

            if self.config.num_feats != 0:
                feature_embs = self.get_feature_embs(
                    ment_start, last_mention_start, ent_counter, metadata
                )
            else:
                feature_embs = torch.empty(mem_vectors.shape[0], 0, device=self.device)

            coref_new_scores = self.get_coref_new_scores(
                ment_emb, mem_vectors, mem_vectors_init, ent_counter, feature_embs
            )

            coref_new_list.append(coref_new_scores)

            # Teacher forcing
            action_str, cell_idx = gt_action_str, gt_cell_idx

            num_ents: int = int(torch.sum((ent_counter > 0).long()).item())
            cell_mask: Tensor = (
                torch.arange(start=0, end=num_ents, device=self.device)
                == torch.tensor(cell_idx)
            ).float()

            mask = torch.unsqueeze(cell_mask, dim=1)
            mask = mask.repeat(1, self.mem_size)

            ## Update memory if action is cluster and memory is not static
            if action_str == "c" and self.config.type != "static":
                coref_vec = self.coref_update(
                    ment_emb, mem_vectors, cell_idx, ent_counter
                )
                mem_vectors = mem_vectors * (1 - mask) + mask * coref_vec
                ent_counter[cell_idx] = ent_counter[cell_idx] + 1
                last_mention_start[cell_idx] = ment_start

        return coref_new_list

    def forward(
        self,
        ment_boundaries: Tensor,
        mention_emb_list: List[Tensor],
        rep_emb_list: List[Tensor],
        gt_actions: List[Tuple[int, str]],
        metadata: Dict,
        teacher_force: False,
        memory_init=None,
    ):
        """Forward pass for clustering entity mentions during inference/evaluation.

        Args:
         ment_boundaries: Start and end token indices for the proposed mentions.
         mention_emb_list: Embedding list of proposed mentions
         metadata: Metadata features such as document genre embedding
         memory_init: Initializer for memory. For streaming coreference, we can pass the previous
                  memory state via this dictionary

        Returns:
                pred_actions: List of predicted clustering actions.
                mem_state: Current memory state.
        """

        ## Check length of mention_emb_list == gt_action
        assert len(mention_emb_list) == len(gt_actions)

        # Initialize memory
        if memory_init is not None:
            mem_vectors, mem_vectors_init, ent_counter, last_mention_start = (
                self.initialize_memory(**memory_init, rep=rep_emb_list)
            )
        else:
            mem_vectors, mem_vectors_init, ent_counter, last_mention_start = (
                self.initialize_memory(rep=rep_emb_list)
            )

        pred_actions = []  # argmax actions
        coref_scores_list = []

        ## Tensorized approach for static method
        if self.config.type == "static":
            batch_size = self.config.batch_size
            ### Mention Emb list gets batched in batch size
            num_batches = len(mention_emb_list) // batch_size + int(
                len(mention_emb_list) % batch_size != 0
            )
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(mention_emb_list))

                num_elements = end_idx - start_idx

                if ent_counter.size() == 0:
                    next_cell_idx, next_action_str = 0, "o"
                    pred_actions.extend(
                        [(next_cell_idx, next_action_str) * num_elements]
                    )
                    continue

                ment_emb_tensor = torch.stack(
                    mention_emb_list[start_idx:end_idx], dim=0
                )
                ment_start, ment_end = (
                    ment_boundaries[start_idx:end_idx, 0],
                    ment_boundaries[start_idx:end_idx, 1],
                )
                if self.config.num_feats != 0:
                    feature_embs = self.get_feature_embs_tensorized(
                        ment_start, last_mention_start, ent_counter, metadata
                    )  ## [B,D,20]
                else:
                    feature_embs = torch.empty(
                        ment_start.shape[0], mem_vectors.shape[0], 0, device=self.device
                    )  ## [B,D,20]
                coref_new_scores = self.get_coref_new_scores_tensorized(
                    ment_emb_tensor,
                    mem_vectors,
                    mem_vectors_init,
                    ent_counter,
                    feature_embs,
                )
                coref_copy = coref_new_scores.clone().detach().cpu()
                coref_scores_list.extend(coref_copy)
                assigned_cluster = self.assign_cluster_tensorized(coref_new_scores)
                gt_actions_batch = gt_actions[start_idx:end_idx]
                if teacher_force:
                    pred_actions.extend(gt_actions_batch)
                else:
                    pred_actions.extend(assigned_cluster)

        else:
            for ment_idx, ment_emb in enumerate(mention_emb_list):

                if ent_counter.size() == 0:
                    next_cell_idx, next_action_str = 0, "o"
                    pred_actions.append((next_cell_idx, next_action_str))
                    continue

                ment_start, ment_end = ment_boundaries[ment_idx]

                if self.config.num_feats != 0:
                    feature_embs = self.get_feature_embs(
                        ment_start, last_mention_start, ent_counter, metadata
                    )
                else:
                    feature_embs = torch.empty(
                        mem_vectors.shape[0], 0, device=self.device
                    )

                coref_new_scores = self.get_coref_new_scores(
                    ment_emb, mem_vectors, mem_vectors_init, ent_counter, feature_embs
                )
                coref_copy = coref_new_scores.clone().detach().cpu()
                coref_scores_list.append(coref_copy)
                pred_cell_idx, pred_action_str = self.assign_cluster(coref_new_scores)

                if teacher_force:
                    next_cell_idx, next_action_str = gt_actions[ment_idx]
                    pred_actions.append(gt_actions[ment_idx])
                else:
                    next_cell_idx, next_action_str = pred_cell_idx, pred_action_str
                    pred_actions.append((pred_cell_idx, pred_action_str))

                if next_action_str == "c":
                    coref_vec = self.coref_update(
                        ment_emb, mem_vectors, next_cell_idx, ent_counter
                    )
                    mem_vectors[next_cell_idx] = coref_vec
                    ent_counter[next_cell_idx] = ent_counter[next_cell_idx] + 1
                    last_mention_start[next_cell_idx] = ment_start

        mem_state = {
            "mem": mem_vectors,
            "mem_init": mem_vectors_init,
            "ent_counter": ent_counter,
            "last_mention_start": last_mention_start,
        }
        return pred_actions, mem_state, coref_scores_list
