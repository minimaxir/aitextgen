from torch.utils.data import DataLoader
from torch.optim import Adam
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.core.memory import get_gpu_memory_map
from tqdm.auto import tqdm
import sys
from transformers import DataCollatorForLanguageModeling


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
        "Prepare optimizer"

        optimizer = Adam(
            [p for n, p in self.model.named_parameters()],
            lr=self.hparams["learning_rate"],
            eps=self.hparams["adam_epsilon"],
        )
        return [optimizer]


class ATGProgressBar(ProgressBarBase):
    """A variant progress bar that works off of steps and prints periodically."""

    def __init__(self, save_every, generate_every, output_dir, n_generate, gpu):
        super().__init__()
        self.save_every = save_every
        self.generate_every = generate_every
        self.output_dir = output_dir
        self.n_generate = n_generate
        self.gpu = gpu
        self.steps = 0
        self.total_loss = 0.0

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
        current_loss = float(trainer.progress_bar_dict["loss"])
        self.steps += 1
        if current_loss == current_loss:  # don't add if current_loss is NaN
            self.total_loss += current_loss

        desc = f"Loss: {current_loss:.3f} — Avg: {self.total_loss / self.steps:.3f}"

        if self.gpu:
            desc += f" — GPU Mem: {get_gpu_memory_map()['gpu_0']} MB"
        self.main_progress_bar.update()
        self.main_progress_bar.set_description(desc)

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
