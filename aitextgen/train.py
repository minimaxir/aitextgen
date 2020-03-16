# coding=utf-8
# Adapted from from the transformers_base.py example from
# huggingface/transformers examples contributed to that repo by Sasha Rush.
#

import logging

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from .transformer_base import BaseTransformer


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
            "labels": batch[1],
        }

        outputs = self.forward(**inputs)
        loss = outputs[0]
        tensorboard_logs = {
            "loss": loss,
            "rate": self.lr_scheduler.get_last_lr()[-1],
        }
        return {"loss": loss, "log": tensorboard_logs}

    def load_dataset(self, mode, batch_size):
        "Load datasets. Called after prepare data."

        def collate(examples):
            if self.tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(
                examples, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )

        if self.hparams.n_gpu > 1 or self.hparams.n_tpu_cores > 0:
            train_sampler = DistributedSampler(self.dataset)
        else:
            train_sampler = RandomSampler(self.dataset)

        return DataLoader(
            self.dataset,
            sampler=train_sampler,
            batch_size=batch_size,
            collate_fn=collate,
        )

