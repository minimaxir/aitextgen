import os
import shutil
import subprocess
import sys

from tqdm.auto import tqdm
from transformers import TrainerCallback


class ATGProgressCallback(TrainerCallback):
    """A variant progress bar that works off of steps and prints periodically."""

    def __init__(self, trainer, num_steps, tokenizer, refresh_rate):
        self.training_bar = None
        self.trainer = trainer
        self.num_steps = num_steps
        self.tokenizer = tokenizer
        self.refresh_rate = refresh_rate

    @property
    def save_every_check(self):
        return self.save_every > 0 and self.steps % self.save_every == 0

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar = tqdm(
                total=self.num_steps,
                smoothing=0,
                leave=True,
                dynamic_ncols=True,
                file=sys.stdout,
            )
        self.current_step = 0

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar.close()
            self.training_bar = None

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.is_local_process_zero:
            self.current_loss = float(metrics.get("train_loss"))

    def on_step_end(self, args, state, control, model, **kwargs):

        if state.is_local_process_zero:
            self.steps += 1
            avg_loss = 0
            if (
                self.current_loss == self.current_loss
            ):  # don't add if current_loss is NaN
                avg_loss = self.average_loss(
                    self.current_loss, self.prev_avg_loss, self.smoothing
                )
                self.prev_avg_loss = avg_loss

            desc = f"Loss: {self.current_loss:.3f} — Avg: {avg_loss:.3f}"

            if state.global_step % self.progress_bar_refresh_rate == 0:
                if self.gpu:
                    # via pytorch-lightning's get_gpu_memory_map()
                    result = subprocess.run(
                        [
                            shutil.which("nvidia-smi"),
                            "--query-gpu=memory.used",
                            "--format=csv,nounits,noheader",
                        ],
                        encoding="utf-8",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True,
                    )
                    gpu_memory = result.stdout.strip().split(os.linesep)[0]
                    desc += f" — GPU Mem: {gpu_memory} MB"
                self.main_progress_bar.update(self.progress_bar_refresh_rate)
                self.main_progress_bar.set_description(desc)

            if _TPU_AVAILABLE and self.save_every_check:
                did_unfreeze = False
                if self.enabled:
                    self.unfreeze_layers(pl_module)
                    did_unfreeze = True
                self.save_pytorch_model(trainer, pl_module, tpu=True)
                if did_unfreeze:
                    self.freeze_layers(pl_module)

            if self.enabled:
                did_unfreeze = False
                if not _TPU_AVAILABLE and self.save_every_check:
                    self.unfreeze_layers(pl_module)
                    self.save_pytorch_model(trainer, pl_module)
                    did_unfreeze = True

                if self.generate_every > 0 and self.steps % self.generate_every == 0:
                    self.unfreeze_layers(pl_module)
                    self.generate_sample_text(trainer, pl_module)
                    did_unfreeze = True

                if did_unfreeze:
                    self.freeze_layers(pl_module)

    def generate_sample_text(self, trainer, pl_module):
        self.main_progress_bar.write(
            f"\033[1m{self.steps:,} steps reached: generating sample texts.\033[0m"
        )

        gen_length_max = getattr(
            pl_module.model.config, "n_positions", None
        ) or getattr(pl_module.model.config, "max_position_embeddings", None)
        gen_length = min(gen_length_max, 256)

        pad_token_id = getattr(pl_module.tokenizer, "pad_token_id", None) or getattr(
            pl_module.tokenizer, "eos_token_id", None
        )

        outputs = pl_module.model.generate(
            input_ids=None,
            max_length=gen_length,
            do_sample=True,
            num_return_sequences=self.n_generate,
            temperature=0.7,
            pad_token_id=pad_token_id,
        )

        gen_texts = pl_module.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for text in gen_texts:
            self.main_progress_bar.write("=" * 10)
            self.main_progress_bar.write(text)

        self.main_progress_bar.write("=" * 10)

    def save_pytorch_model(self, trainer, pl_module, tpu=False):

        if self.enabled:
            self.main_progress_bar.write(
                f"\033[1m{self.steps:,} steps reached: saving model to /{self.output_dir}\033[0m"
            )
        if tpu:
            import torch_xla.core.xla_model as xm

            pl_module.model.save_pretrained(self.output_dir, save_function=xm.save)
        else:
            pl_module.model.save_pretrained(self.output_dir)

        if self.enabled and self.save_gdrive:
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
