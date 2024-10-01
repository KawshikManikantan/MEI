import torch
from model.memory import BaseMemory
import torch.nn as nn

from omegaconf import DictConfig
from typing import Dict, Tuple, List
from torch import Tensor


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
        # Initialize memory
        first_overwrite, coref_new_list = True, []
        mem_vectors, ent_counter, last_mention_start, first_mention_start = (
            self.initialize_memory()
        )

        for ment_idx, (ment_emb, (gt_cell_idx, gt_action_str)) in enumerate(
            zip(mention_emb_list, gt_actions)
        ):
            ment_start, ment_end = ment_boundaries[ment_idx]
            if first_overwrite:
                first_overwrite = False
                mem_vectors = torch.unsqueeze(ment_emb, dim=0)
                ent_counter = torch.tensor([1.0], device=self.device)
                last_mention_start = torch.tensor(
                    [ment_start], dtype=torch.long, device=self.device
                )
                first_mention_start = torch.tensor(
                    [ment_start], dtype=torch.long, device=self.device
                )
                continue
            else:
                if self.config.num_feats != 0:
                    feature_embs = self.get_feature_embs(
                        ment_start, last_mention_start, ent_counter, metadata
                    )
                else:
                    feature_embs = torch.empty(
                        mem_vectors.shape[0], 0, device=self.device
                    )
                coref_new_scores = self.get_coref_new_scores(
                    ment_emb, mem_vectors, ent_counter, feature_embs
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

            if action_str == "c":
                coref_vec = self.coref_update(
                    ment_emb, mem_vectors, cell_idx, ent_counter
                )
                mem_vectors = mem_vectors * (1 - mask) + mask * coref_vec
                ent_counter[cell_idx] = ent_counter[cell_idx] + 1
                last_mention_start[cell_idx] = ment_start
            elif action_str == "o":
                # Append the new entity
                mem_vectors = torch.cat(
                    [mem_vectors, torch.unsqueeze(ment_emb, dim=0)], dim=0
                )
                ent_counter = torch.cat(
                    [ent_counter, torch.tensor([1.0], device=self.device)], dim=0
                )
                last_mention_start = torch.cat(
                    [last_mention_start, ment_start.unsqueeze(dim=0)], dim=0
                )
                first_mention_start = torch.cat(
                    [first_mention_start, ment_start.unsqueeze(dim=0)], dim=0
                )
        return coref_new_list

    def forward(
        self,
        ment_boundaries: Tensor,
        mention_emb_list: Tensor,
        gt_actions: List[Tuple[int, str]],
        metadata: Dict,
        teacher_force: False,
        memory_init: Dict = None,
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
            mem_vectors, ent_counter, last_mention_start, first_mention_start = (
                self.initialize_memory(**memory_init)
            )
        else:
            mem_vectors, ent_counter, last_mention_start, first_mention_start = (
                self.initialize_memory()
            )

        pred_actions = []  # argmax actions
        coref_scores_list = []
        # Boolean to track if we have started tracking any entities
        # This value can be false if we are processing subsequent chunks of a long document
        first_overwrite: bool = True if torch.sum(ent_counter) == 0 else False

        for ment_idx, ment_emb in enumerate(mention_emb_list):
            ment_start, ment_end = ment_boundaries[ment_idx]

            if self.config.num_feats != 0:
                feature_embs = self.get_feature_embs(
                    ment_start, last_mention_start, ent_counter, metadata
                )
            else:
                feature_embs = torch.empty(mem_vectors.shape[0], 0, device=self.device)

            if first_overwrite:
                pred_cell_idx, pred_action_str = 0, "o"
            else:
                coref_new_scores = self.get_coref_new_scores(
                    ment_emb, mem_vectors, ent_counter, feature_embs
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

            if first_overwrite:
                first_overwrite = False
                # We start with a single empty memory cell
                mem_vectors = torch.unsqueeze(ment_emb, dim=0)
                ent_counter = torch.tensor([1.0], device=self.device)
                last_mention_start[0] = ment_start
                first_mention_start[0] = ment_start
            else:
                if next_action_str == "c":
                    # Perform coreference update on the cluster referenced by pred_cell_idx
                    coref_vec = self.coref_update(
                        ment_emb, mem_vectors, next_cell_idx, ent_counter
                    )
                    mem_vectors[next_cell_idx] = coref_vec
                    ent_counter[next_cell_idx] = ent_counter[next_cell_idx] + 1
                    last_mention_start[next_cell_idx] = ment_start

                elif next_action_str == "o":
                    # Append the new entity to the entity cluster array
                    mem_vectors = torch.cat(
                        [mem_vectors, torch.unsqueeze(ment_emb, dim=0)], dim=0
                    )
                    ent_counter = torch.cat(
                        [ent_counter, torch.tensor([1.0], device=self.device)], dim=0
                    )
                    last_mention_start = torch.cat(
                        [last_mention_start, ment_start.unsqueeze(dim=0)], dim=0
                    )
                    first_mention_start = torch.cat(
                        [first_mention_start, ment_start.unsqueeze(dim=0)], dim=0
                    )

        mem_state = {
            "mem": mem_vectors,
            "ent_counter": ent_counter,
            "last_mention_start": last_mention_start,
            "first_mention_start": first_mention_start,
        }
        return pred_actions, mem_state, coref_scores_list
