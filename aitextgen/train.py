# coding=utf-8
#

import logging

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)


class ATGTransformer(pl.LightningModule):
    """
    A training module for aitextgen.
    """

    def __init__(self, model, config, dataset, hparams):
        super(ATGTransformer, self).__init__()
        self.model, self.config, self.dataset, self.hparams = (
            model,
            config,
            dataset,
            hparams,
        )

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

        return {"loss": loss}

    def train_dataloader(self, batch_size):
        "Load datasets. Called after prepare data."

        def collate(examples):
            if self.tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(
                examples, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )

        return DataLoader(self.dataset, batch_size=batch_size, collate_fn=collate,)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=args.num_steps,
        )

        return [optimizer], [scheduler]

    # def optimizer_step(
    #     self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None,
    # ):
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {
            "loss": f"{self.trainer.avg_loss:.3f}",
            "lr": self.lr_scheduler.get_last_lr()[-1],
        }

        return tqdm_dict
