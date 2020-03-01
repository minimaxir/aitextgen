# coding=utf-8
# Adapted from from the transformers_base.py example from
# huggingface/transformers examples contributed to that repo by Sasha Rush.
#


import argparse
import glob
import logging
import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset

from .transformer_base import BaseTransformer, add_generic_args, generic_train


logger = logging.getLogger(__name__)


class ATGTransformer(BaseTransformer):
    """
    A training module for aitextgen. See BaseTransformer for the core options.
    """

    def __init__(self, model, dataset, hparams):
        self.model = model
        self.dataset = dataset
        super(ATGTransformer, self).__init__(hparams)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_num):
        "Compute loss and log."
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[3],
        }
        if self.hparams.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if self.hparams.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use segment_ids

        outputs = self.forward(**inputs)
        loss = outputs[0]
        tensorboard_logs = {
            "loss": loss,
            "rate": self.lr_scheduler.get_last_lr()[-1],
        }
        return {"loss": loss, "log": tensorboard_logs}

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length),
            ),
        )

    def prepare_data(self):
        "Called to initialize data. Use the call to construct features"
        args = self.hparams
        for mode in ["train", "dev", "test"]:
            cached_features_file = self._feature_file(mode)
            if not os.path.exists(cached_features_file):
                logger.info("Creating features from dataset file at %s", args.data_dir)
                examples = read_examples_from_file(args.data_dir, mode)
                features = convert_examples_to_features(
                    examples,
                    self.labels,
                    args.max_seq_length,
                    self.tokenizer,
                    cls_token_at_end=bool(args.model_type in ["xlnet"]),
                    cls_token=self.tokenizer.cls_token,
                    cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                    sep_token=self.tokenizer.sep_token,
                    sep_token_extra=bool(args.model_type in ["roberta"]),
                    pad_on_left=bool(args.model_type in ["xlnet"]),
                    pad_token=self.tokenizer.convert_tokens_to_ids(
                        [self.tokenizer.pad_token]
                    )[0],
                    pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                    pad_token_label_id=self.pad_token_label_id,
                )
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

    def load_dataset(self, mode, batch_size):
        "Load datasets. Called after prepare data."

        def collate(examples: List[torch.Tensor]):
            if tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(
                examples, batch_first=True, padding_value=tokenizer.pad_token_id
            )

        return DataLoader(
            self.train_dataset, batch_size=train_batch_size, collate_fn=collate,
        )

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        # Add NER specific options
        BaseTransformer.add_model_specific_args(parser, root_dir)
        parser.add_argument(
            "--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--labels",
            default="",
            type=str,
            help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
        )

        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
        )

        parser.add_argument(
            "--overwrite_cache",
            action="store_true",
            help="Overwrite the cached training and evaluation sets",
        )

        return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = ATGTransformer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    model = ATGTransformer(model, dataset, args)
    trainer = generic_train(model, args)

