import sys
import os
import time
import logging
import torch

## Uncomment the following line to make the code deterministic and use CUBLAS_WORKSPACE_CONFIG=:4096:8
# torch.use_deterministic_algorithms(True)
import json
import numpy as np
import random
import wandb
from time import sleep

from omegaconf import OmegaConf, open_dict, DictConfig
from os import path
from collections import OrderedDict, defaultdict
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer

from data_utils.utils import load_dataset, load_eval_dataset
import pytorch_utils.utils as utils
from torch.profiler import profile, record_function, ProfilerActivity

from model.entity_ranking_model import EntityRankingModel
from data_utils.tensorize_dataset import TensorizeDataset
from pytorch_utils.optimization_utils import get_inverse_square_root_decay

from utils_evaluate import coref_evaluation

from typing import Dict, Union, List, Optional
import copy

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger()

loss_acc_template_dict = {
    "total": 0.0,
    "entity": 0.0,
    "bounded": 0.0,
    "coref": 0.0,
    "mention_count": 0.0,
    "processed_docs": 0.0,
    "ment_correct": 0.0,
    "ment_total": 0.0,
    "ment_tp": 0.0,
    "ment_pp": 0.0,
    "ment_ap": 0.0,
}


class Experiment:
    """Class for training and evaluating coreference models."""

    def __init__(self, config: DictConfig):
        self.config = config

        ## Important Config Information

        print("Seeded: ", config.seed)
        print("Cuda device: ", config.device)
        print("Cuda Available Status: ", torch.cuda.is_available())
        print(
            "A small pause here to check for important config (check while training)..."
        )

        time.sleep(10)

        # Whether to train or not
        self.eval_model: bool = not self.config.train

        # Initialize dictionary to track key training variables
        self.train_info = {
            "val_perf": 0.0,
            "global_steps": 0,
            "num_stuck_evals": 0,
            "peak_memory": 0.0,
        }

        self.wandbdata = {}

        # Initialize model path attributes
        self.model_path = self.config.paths.model_path
        self.best_model_path = self.config.paths.best_model_path

        if not self.eval_model:
            # Step 1 - Initialize model
            self._build_model()
            # Step 2 - Load Data - Data processing choices such as tokenizer will depend on the model
            self._load_data()

            # Step 4 - Loading the checkpoint also restores the training metadata
            self._load_previous_checkpoint()

            # Step 3 - Resume training
            self._setup_training()

            # All set to resume training
            # But first check if training is remaining
            if self._is_training_remaining():
                self.train()

        # Perform final evaluation
        if path.exists(self.best_model_path):
            # Step 1 - Initialize model
            self._initialize_best_model()
            # Step 2 - Load evaluation data
            self._load_data()
            # Step 3 - Perform evaluation
            self.perform_final_eval()
        else:
            logger.info("No model accessible!")
            sys.exit(1)

    def _build_model(self) -> None:
        """Constructs the model with given config."""

        model_params: DictConfig = self.config.model
        train_config: DictConfig = self.config.trainer

        self.model = EntityRankingModel(
            model_config=model_params, train_config=train_config
        )

        if torch.cuda.is_available():
            self.model.cuda(device=self.config.device)
            # print(torch.cuda.memory_summary())

        # Print model
        utils.print_model_info(self.model)
        sys.stdout.flush()

    def _load_data(self):
        """Loads and processes the training and evaluation data.

        Loads the data concerning all the specified datasets for training and eval.
        The first part of this method loads all the data from the preprocessed jsonline files.
        In the second half, the loaded data is tensorized for consumption by the model.

        Apart from loading and processing the data, the method also populates important
        attributes such as:
                num_train_docs_map (dict): Dictionary to maintain the number of training
                        docs per dataset which is useful for implementing sampling in joint training.
                num_training_steps (int): Number of total training steps.
                eval_per_k_steps (int): Number of gradient updates before each evaluation.
        """

        self.data_iter_map, self.conll_data_dir, self.num_split_docs_map = (
            {},
            {},
            {"train": {}, "dev": {}, "test": {}},
        )
        raw_data_map = {}

        max_segment_len: int = self.config.model.doc_encoder.transformer.max_segment_len
        # print("Max Segment Length::::::",max_segment_len)
        model_name: str = self.config.model.doc_encoder.transformer.name
        add_speaker_tokens: bool = self.config.model.doc_encoder.add_speaker_tokens
        base_data_dir: str = path.abspath(self.config.paths.base_data_dir)

        # Load data
        for dataset_name, attributes in self.config.datasets.items():
            num_train_docs: Optional[int] = attributes.get("num_train_docs", None)
            num_dev_docs: Optional[int] = attributes.get("num_dev_docs", None)
            num_test_docs: Optional[int] = attributes.get("num_test_docs", None)
            singleton_file: Optional[str] = attributes.get("singleton_file", None)
            external_md_file: Optional[str] = attributes.get("external_md_file", None)

            if singleton_file is not None:
                singleton_file = path.join(base_data_dir, singleton_file)
                if path.exists(singleton_file):
                    logger.info(f"Singleton file found: {singleton_file}")

            print("External Md File: ", external_md_file)
            if external_md_file is not None:
                external_md_file = path.join(base_data_dir, external_md_file)
                if path.exists(external_md_file):
                    logger.info(
                        f"External mention detector file found: {external_md_file}"
                    )

            # Data directory is a function of dataset name and tokenizer used
            data_dir = path.join(path.join(base_data_dir, dataset_name), model_name)
            # Check if speaker tokens are added
            if add_speaker_tokens:
                pot_data_dir = path.join(
                    path.join(path.join(base_data_dir, dataset_name)),
                    model_name + "_speaker",
                )
                if path.exists(pot_data_dir):
                    data_dir = pot_data_dir

            # Datasets such as litbank have cross validation splits
            if attributes.get("cross_val_split", None) is not None:
                data_dir = path.join(data_dir, str(attributes.get("cross_val_split")))

            logger.info("Data directory: %s" % data_dir)

            # CoNLL data dir
            if attributes.get("has_conll", False):
                conll_dir = path.join(
                    path.join(path.join(base_data_dir, dataset_name)), "conll"
                )
                if attributes.get("cross_val_split", None) is not None:
                    # LitBank like datasets have cross validation splits
                    conll_dir = path.join(
                        conll_dir, str(attributes.get("cross_val_split"))
                    )

                if path.exists(conll_dir):
                    self.conll_data_dir[dataset_name] = conll_dir

            self.num_split_docs_map["train"][dataset_name] = num_train_docs
            self.num_split_docs_map["dev"][dataset_name] = num_dev_docs
            self.num_split_docs_map["test"][dataset_name] = num_test_docs

            if self.eval_model:
                print("In Eval Model DataLoader")
                raw_data_map[dataset_name] = load_eval_dataset(
                    data_dir,
                    external_md_file=external_md_file,
                    max_segment_len=max_segment_len,
                    dataset_name=dataset_name,
                )
            else:
                raw_data_map[dataset_name] = load_dataset(
                    data_dir,
                    singleton_file=singleton_file,
                    num_dev_docs=num_dev_docs,
                    num_test_docs=num_test_docs,
                    max_segment_len=max_segment_len,
                    dataset_name=dataset_name,
                )

        # Tensorize data
        data_processor = TensorizeDataset(
            self.model.get_tokenizer(),
            remove_singletons=(not self.config.keep_singletons),
        )

        if self.eval_model:
            for split in ["dev", "test"]:
                self.data_iter_map[split] = {}

            for dataset in raw_data_map:
                for split in raw_data_map[dataset]:
                    self.data_iter_map[split][dataset] = data_processor.tensorize_data(
                        raw_data_map[dataset][split], training=False
                    )
        else:
            # Training
            for split in ["train", "dev", "test"]:
                self.data_iter_map[split] = {}
                training = split == "train"
                for dataset in raw_data_map:
                    self.data_iter_map[split][dataset] = data_processor.tensorize_data(
                        raw_data_map[dataset][split], training=training
                    )

            # Estimate number of training steps
            if self.config.trainer.eval_per_k_steps is None:
                # Eval steps is 1 epoch (with subsampling) of all the datasets used in joint training
                self.config.trainer.eval_per_k_steps = sum(
                    self.num_split_docs_map["train"].values()
                )

            self.config.trainer.num_training_steps = (
                self.config.trainer.eval_per_k_steps * self.config.trainer.max_evals
            )
            logger.info(
                f"Number of training steps: {self.config.trainer.num_training_steps}"
            )

            logger.info(f"Eval per k steps: {self.config.trainer.eval_per_k_steps}")

    def _load_previous_checkpoint(self):
        """Loads the last checkpoint or best checkpoint."""

        # Resume training
        if path.exists(self.model_path):
            self.load_model(self.model_path, last_checkpoint=True)
            logger.info("Model loaded\n")
        else:
            # Starting training

            logger.info("Model initialized\n")
            sys.stdout.flush()

    def _is_training_remaining(self):
        """Check if training is done or remaining.

        There are two cases where we don't resume training:
        (a) The dev performance has not improved for the allowed patience parameter number of evaluations.
        (b) Number of gradient updates is already >= Total training steps.

        Returns:
                bool: If true, we resume training. Otherwise do final evaluation.
        """

        if self.train_info["num_stuck_evals"] >= self.config.trainer.patience:
            return False
        if self.train_info["global_steps"] >= self.config.trainer.num_training_steps:
            return False

        return True

    def _setup_training(self):
        """Initialize optimizer and bookkeeping variables for training."""

        # Dictionary to track key training variables
        self.train_info = {
            "val_perf": 0.0,
            "global_steps": 0,
            "num_stuck_evals": 0,
            "peak_memory": 0.0,
            "max_mem": 0.0,
        }

        # Initialize optimizers
        self._initialize_optimizers()

    def _initialize_optimizers(self):
        """Initialize model + optimizer(s). Check if there's a checkpoint in which case we resume from there."""

        optimizer_config: DictConfig = self.config.optimizer
        train_config: DictConfig = self.config.trainer
        self.optimizer, self.optim_scheduler = {}, {}

        if torch.cuda.is_available():
            # Gradient scaler required for mixed precision training
            self.scaler = torch.GradScaler("cuda")
        else:
            self.scaler = None

        # Optimizer for clustering params
        self.optimizer["mem"] = torch.optim.Adam(
            self.model.get_params()[1], lr=optimizer_config.init_lr, eps=1e-6
        )

        if optimizer_config.lr_decay == "inv":
            self.optim_scheduler["mem"] = get_inverse_square_root_decay(
                self.optimizer["mem"], num_warmup_steps=0
            )
        else:
            # No warmup steps for model params
            self.optim_scheduler["mem"] = get_linear_schedule_with_warmup(
                self.optimizer["mem"],
                num_warmup_steps=0,
                num_training_steps=train_config.num_training_steps,
            )

        if self.config.model.doc_encoder.finetune:
            # Optimizer for document encoder
            no_decay = [
                "bias",
                "LayerNorm.weight",
            ]  # No weight decay for bias and layernorm weights
            encoder_params = self.model.get_params(named=True)[0]
            grouped_param = [
                {
                    "params": [
                        p
                        for n, p in encoder_params
                        if not any(nd in n for nd in no_decay)
                    ],
                    "lr": optimizer_config.fine_tune_lr,
                    "weight_decay": 1e-2,
                },
                {
                    "params": [
                        p for n, p in encoder_params if any(nd in n for nd in no_decay)
                    ],
                    "lr": optimizer_config.fine_tune_lr,
                    "weight_decay": 0.0,
                },
            ]

            self.optimizer["doc"] = torch.optim.AdamW(
                grouped_param, lr=optimizer_config.fine_tune_lr, eps=1e-6
            )

            # Scheduler for document encoder
            num_warmup_steps = int(0.1 * train_config.num_training_steps)
            if optimizer_config.lr_decay == "inv":
                self.optim_scheduler["doc"] = get_inverse_square_root_decay(
                    self.optimizer["doc"], num_warmup_steps=num_warmup_steps
                )
            else:
                self.optim_scheduler["doc"] = get_linear_schedule_with_warmup(
                    self.optimizer["doc"],
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=train_config.num_training_steps,
                )
            # print("Original: ",self.optim_scheduler["doc"].state_dict())

    def agg(self, datadepdict):
        agg_dict = defaultdict(float)
        print(datadepdict)
        for dataset in datadepdict:
            for key in datadepdict[dataset]:
                agg_dict[key] += datadepdict[dataset][key]

        # for dataset in datadepdict:
        #     datadepdict[dataset]["loss_norm"] = datadepdict[dataset]["total"]/datadepdict[dataset]["mention_count"]

        agg_dict["loss_norm"] = (
            agg_dict["coref"] / agg_dict["mention_count"]
            + agg_dict["entity"] / agg_dict["ment_total"]
        )
        agg_dict["ment_acc"] = agg_dict["ment_correct"] / agg_dict["ment_total"]
        agg_dict["ment_prec"] = (
            agg_dict["ment_tp"] / agg_dict["ment_pp"] if agg_dict["ment_pp"] > 0 else 0
        )
        agg_dict["ment_rec"] = (
            agg_dict["ment_tp"] / agg_dict["ment_ap"] if agg_dict["ment_ap"] > 0 else 0
        )
        agg_dict["ment_f1"] = (
            2
            * (agg_dict["ment_prec"] * agg_dict["ment_rec"])
            / (agg_dict["ment_prec"] + agg_dict["ment_rec"])
            if (agg_dict["ment_prec"] + agg_dict["ment_rec"]) > 0
            else 0
        )

        return agg_dict

    def train(self) -> None:
        """Method for training the model.

        This method implements the training loop.
        Within the training loop, the model is periodically evaluated on the dev set(s).
        """

        model, optimizer, scheduler, scaler = (
            self.model,
            self.optimizer,
            self.optim_scheduler,
            self.scaler,
        )
        model.train()

        optimizer_config, train_config = self.config.optimizer, self.config.trainer

        start_time = time.time()
        # fscore = self.periodic_model_eval()
        eval_time = {"total_time": 0, "num_evals": 0}
        print("Started Training..")
        while True:
            logger.info("Steps done %d" % (self.train_info["global_steps"]))
            train_data = self.runtime_load_dataset("train")
            logger.info("Per epoch training steps: %d" % len(train_data))
            logger.info("Is Torch Available: %s" % torch.cuda.is_available())

            # Shuffle the concatenated examples again
            np.random.shuffle(train_data)
            encoder_params, task_params = model.get_params()
            stat_per_dataset = defaultdict(
                lambda: copy.deepcopy(loss_acc_template_dict)
            )
            agg_stat = self.agg

            # Training "epoch" -> May not correspond to actual epoch
            for cur_document in train_data:

                def handle_example(document: Dict) -> Union[None, float]:
                    self.train_info["global_steps"] += 1
                    for key in optimizer:
                        optimizer[key].zero_grad()

                    loss_dict: Dict = model.forward_training(document)
                    total_loss = loss_dict["total"]
                    if total_loss is None or torch.isnan(total_loss):
                        print("Problem with Loss. Should not occur often")
                        return None

                    total_loss.backward()

                    # Gradient clipping
                    try:
                        for name_ind, param_group in enumerate(
                            [encoder_params, task_params]
                        ):
                            torch.nn.utils.clip_grad_norm_(
                                param_group,
                                optimizer_config.max_gradient_norm,
                                error_if_nonfinite=True,
                            )
                    except RuntimeError:
                        print("Non Finite Gradient")
                        return None

                    for key in optimizer:
                        self.wandbdata[key + "_lr"] = scheduler[key].get_last_lr()[0]

                    for key in optimizer:
                        optimizer[key].step()
                        scheduler[key].step()

                    loss_dict_items = {}
                    for key in loss_dict:
                        loss_dict_items[key] = loss_dict[key].item()

                    dataset_name = document["dataset_name"]

                    for key in loss_dict_items:
                        stat_per_dataset[dataset_name][key] += loss_dict_items[key]

                    stat_per_dataset[dataset_name]["processed_docs"] += 1

                    return total_loss.item()

                loss = handle_example(cur_document)

                if self.train_info["global_steps"] % train_config.log_frequency == 0:
                    max_mem = (
                        (
                            torch.cuda.max_memory_allocated(self.config.device)
                            / (1024**3)
                        )
                        if torch.cuda.is_available()
                        else 0.0
                    )
                    if self.train_info.get("max_mem", 0.0) < max_mem:
                        self.train_info["max_mem"] = max_mem

                    if loss is not None:
                        logger.info(
                            "{} {:.3f} Max mem {:.1f} GB".format(
                                cur_document["doc_key"],
                                loss,
                                max_mem,
                            )
                        )
                    sys.stdout.flush()
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats(self.config.device)

                if train_config.eval_per_k_steps and (
                    self.train_info["global_steps"] % train_config.eval_per_k_steps == 0
                ):
                    coref_dict = {}
                    print("Train stats of epoch: ", stat_per_dataset)
                    if self.config.use_wandb:
                        self._wandb_log(
                            split="train",
                            stat_per_dataset=stat_per_dataset,
                            agg_stat=agg_stat,
                            coref_dict=coref_dict,
                            step=self.train_info["global_steps"]
                            // train_config.eval_per_k_steps,
                        )

                    stat_per_dataset = defaultdict(
                        lambda: copy.deepcopy(loss_acc_template_dict)
                    )

                    fscore = self.periodic_model_eval()

                    model.train()
                    # Get elapsed time
                    elapsed_time = time.time() - start_time

                    start_time = time.time()
                    logger.info(
                        "Steps: %d, F1: %.1f, Max F1: %.1f, Time: %.2f"
                        % (
                            self.train_info["global_steps"],
                            fscore,
                            self.train_info["val_perf"],
                            elapsed_time,
                        )
                    )

                    # Check stopping criteria
                    if not self._is_training_remaining():
                        break

            # Check stopping criteria
            if not self._is_training_remaining():
                break

            logger.handlers[0].flush()

    def runtime_load_dataset(self, split):
        # Shuffle and load the training data
        data = []
        for dataset, dataset_data in self.data_iter_map[split].items():
            np.random.shuffle(dataset_data)
            if self.num_split_docs_map[split].get(dataset, None) is not None:
                # Subsampling the data - This is useful in joint training
                logger.info(
                    f"{dataset}: Subsampled {self.num_split_docs_map[split].get(dataset)}"
                )
                random_indices = range(self.num_split_docs_map[split].get(dataset))
                data += [dataset_data[idx] for idx in random_indices]
            else:
                data += dataset_data
        return data

    def _wandb_log(self, split, stat_per_dataset, agg_stat, coref_dict, step=None):
        for dataset_name in stat_per_dataset:
            for metric_vals in stat_per_dataset[dataset_name]:
                print(f"{split}/{dataset_name}/{metric_vals}")
                wandb.log(
                    data={
                        f"{split}/{dataset_name}/{metric_vals}": stat_per_dataset[
                            dataset_name
                        ][metric_vals]
                    },
                    step=step,
                )
            if stat_per_dataset[dataset_name]["mention_count"] > 0.0:
                ment_prec = (
                    stat_per_dataset[dataset_name]["ment_tp"]
                    / stat_per_dataset[dataset_name]["ment_pp"]
                    if stat_per_dataset[dataset_name]["ment_pp"] > 0
                    else 0
                )
                ment_rec = (
                    stat_per_dataset[dataset_name]["ment_tp"]
                    / stat_per_dataset[dataset_name]["ment_ap"]
                    if stat_per_dataset[dataset_name]["ment_ap"] > 0
                    else 0
                )
                ment_f1 = (
                    2 * (ment_prec * ment_rec) / (ment_prec + ment_rec)
                    if (ment_prec + ment_rec) > 0
                    else 0
                )
                wandb.log(
                    data={
                        f"{split}/{dataset_name}/loss_norm": stat_per_dataset[
                            dataset_name
                        ]["coref"]
                        / stat_per_dataset[dataset_name]["mention_count"]
                        + stat_per_dataset[dataset_name]["entity"]
                        / stat_per_dataset[dataset_name]["ment_total"],
                        f"{split}/{dataset_name}/ment_acc": stat_per_dataset[
                            dataset_name
                        ]["ment_correct"]
                        / stat_per_dataset[dataset_name]["ment_total"],
                        f"{split}/{dataset_name}/ment_prec": ment_prec,
                        f"{split}/{dataset_name}/ment_rec": ment_rec,
                        f"{split}/{dataset_name}/ment_f1": ment_f1,
                    },
                    step=step,
                )
            else:
                print("No mentions processed. Should not occur many times.")

        if agg_stat:
            for metric in agg_stat(stat_per_dataset):
                wandb.log(
                    data={f"{split}/{metric}": agg_stat(stat_per_dataset)[metric]},
                    step=step,
                )

        for dataset in coref_dict:
            for key in coref_dict[dataset]:
                # Log result for individual metrics
                if isinstance(coref_dict[dataset][key], dict):
                    wandb.log(
                        data={
                            f"{split}/{dataset}/{key}": coref_dict[dataset][key].get(
                                "fscore", 0.0
                            )
                        },
                        step=step,
                    )

            # Log the overall F-score
            wandb.log(
                data={
                    f"{split}/{dataset}/CoNLL": coref_dict[dataset].get("fscore", 0.0)
                },
                step=step,
            )

        wandb.log(data=self.wandbdata, step=step)

    @torch.no_grad()
    def periodic_model_eval(self) -> float:
        """Method for evaluating and saving the model during the training loop.

        Returns:
                float: Average CoNLL F-score over all the development sets of datasets.
        """

        self.model.eval()

        ## Dev Loss Calculations:
        dev_data = self.runtime_load_dataset("dev")
        np.random.shuffle(dev_data)
        stat_per_dataset = defaultdict(lambda: copy.deepcopy(loss_acc_template_dict))
        agg_stat = self.agg

        for cur_document in dev_data:

            def handle_example(document: Dict) -> Union[None, float]:
                loss_dict: Dict = self.model.forward_training(document)
                total_loss = loss_dict["total"]
                if total_loss is None or torch.isnan(total_loss):
                    print("Problem with Loss. Should not occur many times")
                    return None

                loss_dict_items = {}
                for key in loss_dict:
                    loss_dict_items[key] = loss_dict[key].item()

                dataset_name = document["dataset_name"]

                for key in loss_dict_items:
                    stat_per_dataset[dataset_name][key] += loss_dict_items[key]

                stat_per_dataset[dataset_name]["processed_docs"] += 1
                return total_loss.item()

            loss = handle_example(cur_document)
            if loss is None:
                continue

        # Dev performance
        coref_dict = {}
        train_config = self.config.trainer
        for dataset in self.data_iter_map["dev"]:
            for go in [False]:
                for tf in [False]:
                    result_dict = coref_evaluation(
                        self.config,
                        self.model,
                        self.data_iter_map,
                        dataset,
                        teacher_force=tf,
                        gold_mentions=go,
                        _iter="_"
                        + str(
                            self.train_info["global_steps"]
                            // train_config.eval_per_k_steps
                        ),
                        conll_data_dir=self.conll_data_dir,
                    )

                    coref_dict[dataset] = result_dict

        if self.config.use_wandb:
            self._wandb_log(
                split="dev",
                stat_per_dataset=stat_per_dataset,
                agg_stat=agg_stat,
                coref_dict=coref_dict,
                step=self.train_info["global_steps"] // train_config.eval_per_k_steps,
            )

        fscore = sum([coref_dict[dataset]["fscore"] for dataset in coref_dict]) / len(
            coref_dict
        )
        logger.info("F1: %.1f, Max F1: %.1f" % (fscore, self.train_info["val_perf"]))

        # Update model if dev performance improves
        if fscore > self.train_info["val_perf"]:
            # Update training bookkeeping variables
            self.train_info["num_stuck_evals"] = 0
            self.train_info["val_perf"] = fscore

            # Save the best model
            logger.info("Saving best model")
            self.save_model(self.best_model_path, last_checkpoint=False)
        else:
            self.train_info["num_stuck_evals"] += 1

        # Save model
        if self.config.trainer.to_save_model:
            self.save_model(self.model_path, last_checkpoint=True)

        # Go back to training mode
        self.model.train()
        return fscore

    @torch.no_grad()
    def perform_final_eval(self) -> None:
        """Method to evaluate the model after training has finished."""

        self.model.eval()
        base_output_dict = OmegaConf.to_container(self.config)
        perf_summary = {"best_perf": self.train_info["val_perf"]}
        if self.config.paths.model_dir:
            perf_summary["model_dir"] = path.normpath(self.config.paths.model_dir)

        logger.info(
            "Max training memory: %.1f GB" % self.train_info.get("max_mem", 0.0)
        )

        logger.info("Validation performance: %.1f" % self.train_info["val_perf"])

        perf_file_dict = {}
        dataset_output_dict = {}

        for split in ["dev", "test"]:
            perf_summary[split] = {}
            logger.info("\n")
            logger.info("%s" % split.capitalize())
            coref_dict = {}
            for dataset in self.data_iter_map.get(split, []):
                dataset_dir = path.join(self.config.paths.model_dir, dataset)
                if not path.exists(dataset_dir):
                    os.makedirs(dataset_dir)

                if dataset not in dataset_output_dict:
                    dataset_output_dict[dataset] = {}
                if dataset not in perf_file_dict:
                    perf_file_dict[dataset] = path.join(dataset_dir, f"perf.json")

                print("Dataset Name:", self.config.datasets[dataset].name)
                logger.info("Dataset: %s\n" % self.config.datasets[dataset].name)

                # print("Data Iter map:", self.data_iter_map)
                print("Dataset:", dataset)
                print("Split:", split)
                for go in [False]:
                    for tf in [False]:
                        result_dict = coref_evaluation(
                            self.config,
                            self.model,
                            self.data_iter_map,
                            dataset=dataset,
                            split=split,
                            teacher_force=tf,
                            gold_mentions=go,
                            final_eval=True,
                            conll_data_dir=self.conll_data_dir,
                        )
                        coref_dict[dataset] = result_dict
                        dataset_output_dict[dataset][split] = result_dict
                        perf_summary[split][dataset] = result_dict["fscore"]

            if self.config.use_wandb:
                self._wandb_log(
                    split=split,
                    stat_per_dataset={},
                    agg_stat=None,
                    coref_dict=coref_dict,
                    step=None,
                )

            sys.stdout.flush()

        for dataset, output_dict in dataset_output_dict.items():
            perf_file = perf_file_dict[dataset]
            json.dump(output_dict, open(perf_file, "w"), indent=2)
            logger.info("Final performance summary at %s" % path.abspath(perf_file))

        summary_file = path.join(self.config.paths.model_dir, "perf.json")
        json.dump(perf_summary, open(summary_file, "w"), indent=2)
        logger.info("Performance summary file: %s" % path.abspath(summary_file))

    def _initialize_best_model(self):
        checkpoint = torch.load(
            self.best_model_path, map_location="cpu", weights_only=False
        )
        config = checkpoint["config"]
        # Copying the saved model config to current config is very important to avoid any issues while
        # loading the saved model. To give an example, model might be saved with the speaker tags
        # (training: experiment=ontonotes_speaker)
        # but the evaluation config might lack this detail (eval: experiment=eval_all)
        # However, overriding the encoder is possible -- This method is a bit hacky but allows for overriding the pretrained
        # transformer model from command line.
        if self.config.get("override_encoder", False):
            model_config = config.model
            print(type(self.config.model.doc_encoder.transformer))
            print(self.config.model.doc_encoder.transformer)
            model_config.doc_encoder.transformer = (
                self.config.model.doc_encoder.transformer
            )

        # Override memory
        # For e.g., can test with a different bounded memory size
        if self.config.get("override_memory", False):
            model_config = config.model
            model_config.memory = self.config.model.memory

        with open_dict(config):
            config.model.mention_params.ext_ment = (
                self.config.model.mention_params.ext_ment
            )

        self.config.model = config.model

        self.train_info = checkpoint["train_info"]

        if self.config.model.doc_encoder.finetune:
            # Load the document encoder params if encoder is finetuned
            doc_encoder_dir = path.join(
                path.dirname(self.best_model_path),
                self.config.paths.doc_encoder_dirname,
            )
            if path.exists(doc_encoder_dir):
                logger.info(
                    "Loading document encoder from %s" % path.abspath(doc_encoder_dir)
                )
                config.model.doc_encoder.transformer.model_str = doc_encoder_dir

        self.model = EntityRankingModel(config.model, config.trainer)
        # Document encoder parameters will be loaded via the huggingface initialization
        self.model.load_state_dict(checkpoint["model"], strict=False)
        if torch.cuda.is_available():
            self.model.cuda(device=self.config.device)

    def load_model(self, location: str, last_checkpoint=True) -> None:
        """Load model from given location.

        Args:
                location: str
                        Location of checkpoint
                last_checkpoint: bool
                        Whether the checkpoint is the last one saved or not.
                        If false, don't load optimizers, schedulers, and other training variables.
        """

        checkpoint = torch.load(location, map_location="cpu", weights_only=False)
        logger.info("Loading model from %s" % path.abspath(location))

        self.config = checkpoint["config"]
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.train_info = checkpoint["train_info"]

        if self.config.model.doc_encoder.finetune:
            # Load the document encoder params if encoder is finetuned
            doc_encoder_dir = path.join(
                path.dirname(location), self.config.paths.doc_encoder_dirname
            )
            logger.info(
                "Loading document encoder from %s" % path.abspath(doc_encoder_dir)
            )

            # Load the encoder
            self.model.mention_proposer.doc_encoder.lm_encoder = (
                AutoModel.from_pretrained(pretrained_model_name_or_path=doc_encoder_dir)
            )
            self.model.mention_proposer.doc_encoder.tokenizer = (
                AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=doc_encoder_dir,
                    clean_up_tokenization_spaces=True,
                )
            )

            if torch.cuda.is_available():
                self.model.cuda(device=self.config.device)

        if last_checkpoint:  ## Code from longdoc; Not tested for continual training
            # If resuming training, restore the optimizer state as well
            for param_group in checkpoint["optimizer"]:
                self.optimizer[param_group].load_state_dict(
                    checkpoint["optimizer"][param_group]
                )
                self.optim_scheduler[param_group].load_state_dict(
                    checkpoint["scheduler"][param_group]
                )

            if "scaler" in checkpoint and self.scaler is not None:
                self.scaler.load_state_dict(checkpoint["scaler"])

            torch.set_rng_state(checkpoint["rng_state"])
            np.random.set_state(checkpoint["np_rng_state"])

    def save_model(self, location: os.PathLike, last_checkpoint=True) -> None:
        """Save model.

        Args:
                location: Location of checkpoint
                last_checkpoint:
                        Whether the checkpoint is the last one saved or not.
                        If false, don't save optimizers and schedulers which take up a lot of space.
        """

        model_state_dict = OrderedDict(self.model.state_dict())
        doc_encoder_state_dict = {}

        # Separate the doc_encoder state dict
        # We will save the model in two parts:
        # (a) Doc encoder parameters - Useful for final upload to huggingface
        # (b) Rest of the model parameters, optimizers, schedulers, and other bookkeeping variables
        for key in self.model.state_dict():
            if "lm_encoder." in key:
                doc_encoder_state_dict[key] = model_state_dict[key]
                del model_state_dict[key]

        # Save the document encoder params
        if self.config.model.doc_encoder.finetune:
            doc_encoder_dir = path.join(
                path.dirname(location), self.config.paths.doc_encoder_dirname
            )
            if not path.exists(doc_encoder_dir):
                os.makedirs(doc_encoder_dir)

            logger.info(f"Encoder saved at {path.abspath(doc_encoder_dir)}")
            # Save the encoder
            self.model.mention_proposer.doc_encoder.lm_encoder.save_pretrained(
                save_directory=doc_encoder_dir, save_config=True
            )
            # Save the tokenizer
            self.model.mention_proposer.doc_encoder.tokenizer.save_pretrained(
                doc_encoder_dir
            )

        save_dict = {
            "train_info": self.train_info,
            "model": model_state_dict,
            "rng_state": torch.get_rng_state(),
            "np_rng_state": np.random.get_state(),
            "config": self.config,
        }

        if self.scaler is not None:
            save_dict["scaler"] = self.scaler.state_dict()

        if last_checkpoint:
            # For last checkpoint save the optimizer and scheduler states as well
            save_dict["optimizer"] = {}
            save_dict["scheduler"] = {}

            param_groups: List[str] = (
                ["mem", "doc"] if self.config.model.doc_encoder.finetune else ["mem"]
            )
            for param_group in param_groups:
                save_dict["optimizer"][param_group] = self.optimizer[
                    param_group
                ].state_dict()
                save_dict["scheduler"][param_group] = self.optim_scheduler[
                    param_group
                ].state_dict()

        torch.save(save_dict, location)
        logger.info(f"Model saved at: {path.abspath(location)}")
