from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.core.memory import get_gpu_memory_map
from tqdm.auto import tqdm
import sys
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import os
import shutil

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    pass


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

        outputs = self({"input_ids": batch, "labels": batch})
        loss = outputs[0]

        return {"loss": loss, "log": {"Loss": loss}}

    def train_dataloader(self):
        "Load datasets. Called after prepare data."

        return DataLoader(
            self.dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            pin_memory=self.hparams["pin_memory"],
            num_workers=self.hparams["num_workers"],
        )

    def configure_optimizers(self):
        "Prepare optimizer"

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
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

        self.opt = optimizer
        self.lr_scheduler = scheduler
        return [optimizer], [scheduler]

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None
    ):
        if self.hparams["tpu"]:
            xm.optimizer_step(optimizer, barrier=True)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()


class ATGProgressBar(ProgressBarBase):
    """A variant progress bar that works off of steps and prints periodically."""

    def __init__(
        self,
        save_every,
        generate_every,
        output_dir,
        n_generate,
        gpu,
        smoothing,
        run_id,
        save_gdrive,
        progress_bar_refresh_rate,
    ):
        super().__init__()
        self.enabled = True
        self.save_every = save_every
        self.generate_every = generate_every
        self.output_dir = output_dir
        self.n_generate = n_generate
        self.gpu = gpu
        self.steps = 0
        self.prev_avg_loss = None
        self.smoothing = smoothing
        self.run_id = run_id
        self.save_gdrive = save_gdrive
        self.progress_bar_refresh_rate = progress_bar_refresh_rate

    def enabled(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.main_progress_bar = tqdm(
            total=trainer.max_steps,
            disable=not self.enabled,
            smoothing=0,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )

    def on_batch_end(self, trainer, pl_module):
        super().on_batch_end(trainer, pl_module)

        # clean up the GPU cache used for the benchmark
        # https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232/4
        if self.steps == 0 and self.gpu:
            torch.cuda.empty_cache()

        current_loss = float(trainer.progress_bar_dict["loss"])
        self.steps += 1
        avg_loss = 0
        if current_loss == current_loss:  # don't add if current_loss is NaN
            avg_loss = self.average_loss(
                current_loss, self.prev_avg_loss, self.smoothing
            )
            self.prev_avg_loss = avg_loss

        desc = f"Loss: {current_loss:.3f} — Avg: {avg_loss:.3f}"

        if self.steps % self.progress_bar_refresh_rate == 0:
            if self.gpu:
                desc += f" — GPU Mem: {get_gpu_memory_map()['gpu_0']} MB"
            self.main_progress_bar.update(self.progress_bar_refresh_rate)
            self.main_progress_bar.set_description(desc)

        if self.enabled:

            if self.save_every > 0 and self.steps % self.save_every == 0:
                if pl_module.hparams["tpu"]:
                    xm.rendezvous("save_model")
                self.save_pytorch_model(trainer, pl_module)

            if (
                not pl_module.hparams["tpu"]
                and self.generate_every > 0
                and self.steps % self.generate_every == 0
            ):
                self.generate_sample_text(trainer, pl_module)

    def generate_sample_text(self, trainer, pl_module):
        self.main_progress_bar.write(
            f"\033[1m{self.steps:,} steps reached: generating sample texts.\033[0m"
        )

        gen_length = min(pl_module.model.config.n_positions, 256)

        outputs = pl_module.model.generate(
            max_length=gen_length,
            do_sample=True,
            num_return_sequences=self.n_generate,
            temperature=0.7,
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
            f"\033[1m{self.steps:,} steps reached: saving model to /{self.output_dir}\033[0m"
        )
        pl_module.model.save_pretrained(self.output_dir)

        if self.save_gdrive:
            for pt_file in ["pytorch_model.bin", "config.json"]:
                shutil.copyfile(
                    os.path.join(self.output_dir, pt_file),
                    os.path.join("/content/drive/My Drive/", self.run_id, pt_file),
                )

    def average_loss(self, current_loss, prev_avg_loss, smoothing):
        if prev_avg_loss is None:
            return current_loss
        else:
            return (smoothing * current_loss) + (1 - smoothing) * prev_avg_loss
