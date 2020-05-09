from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import ProgressBarBase
from tqdm.auto import tqdm
import sys
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)


class ATGTransformer(pl.LightningModule):
    """
    A training module for aitextgen.
    """

    def __init__(self, model, dataset, hparams, tokenizer):
        super(ATGTransformer, self).__init__()
        self.model, self.dataset, self.hparams, self.tokenizer = (
            model,
            dataset,
            hparams,
            tokenizer,
        )

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_num):
        "Compute loss and log."

        outputs = self(batch)  # batch is a dict containing w/ "input_ids" and "labels"
        loss = outputs[0]

        return {"loss": loss}

    def train_dataloader(self):
        "Load datasets. Called after prepare data."

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        return DataLoader(
            self.dataset,
            batch_size=self.hparams["batch_size"],
            collate_fn=data_collator.collate_batch,
            shuffle=True,
            pin_memory=self.hparams["pin_memory"],
            num_workers=self.hparams["num_workers"],
        )

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
                "weight_decay": self.hparams["weight_decay"],
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
            lr=self.hparams["learning_rate"],
            eps=self.hparams["adam_epsilon"],
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams["warmup_steps"],
            num_training_steps=self.hparams["num_steps"],
        )

        return [optimizer], [scheduler]


class ATGProgressBar(ProgressBarBase):
    """A variant progress bar that works off of steps and prints periodically."""

    def __init__(self, save_every, generate_every, output_dir, n_generate):
        super().__init__()
        self.save_every = save_every
        self.generate_every = generate_every
        self.output_dir = output_dir
        self.n_generate = n_generate
        self.steps = 0

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.main_progress_bar = tqdm(
            total=trainer.max_steps,
            smoothing=0,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )

    def on_batch_end(self, trainer, pl_module):
        super().on_batch_end(trainer, pl_module)
        self.steps += 1
        self.main_progress_bar.update()
        self.main_progress_bar.set_description(
            f"Loss: {trainer.progress_bar_dict['loss']}"
        )

        if self.save_every > 0 and self.steps % self.save_every == 0:
            self.save_pytorch_model(trainer, pl_module)

        if self.generate_every > 0 and self.steps % self.generate_every == 0:
            self.generate_sample_text(trainer, pl_module)

    def generate_sample_text(self, trainer, pl_module):
        self.main_progress_bar.write(
            f"{self.steps:,} steps reached: generating sample texts."
        )

        gen_length = min(pl_module.model.config.n_positions, 256)

        outputs = pl_module.model.generate(
            max_length=gen_length, do_sample=True, num_return_sequences=self.n_generate
        )
        gen_texts = [
            pl_module.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        for text in gen_texts:
            self.main_progress_bar.write("=" * 10)
            self.main_progress_bar.write(text)

        self.main_progress_bar.write("=" * 10)

    def save_pytorch_model(self, trainer, pl_module):
        self.main_progress_bar.write(
            f"{self.steps:,} steps reached: saving model to {self.output_dir}"
        )
        pl_module.model.save_pretrained(self.output_dir)
