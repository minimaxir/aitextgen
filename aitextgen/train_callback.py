import os
import shutil
import subprocess
import sys

from tqdm.auto import tqdm
from transformers import TrainerCallback


class ATGProgressCallback(TrainerCallback):
    """A variant progress bar that works off of steps and prints periodically."""

    def __init__(
        self,
        model,
        trainer,
        tokenizer,
        refresh_rate,
        save_every,
        generate_every,
        n_generate,
        output_dir,
        save_gdrive,
        avg_loss_smoothing,
        is_gpu_used,
    ):
        self.training_bar = None
        self.model = model
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.refresh_rate = refresh_rate
        self.save_every = save_every
        self.generate_every = generate_every
        self.n_generate = n_generate
        self.output_dir = output_dir
        self.save_gdrive = save_gdrive
        self.smoothing = avg_loss_smoothing
        self.gpu = is_gpu_used
        self.steps = 0
        self.current_loss = None
        self.prev_avg_loss = None

    @property
    def save_every_check(self):
        return self.save_every > 0 and self.steps % self.save_every == 0

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar = tqdm(
                total=state.max_steps,
                smoothing=0,
                leave=True,
                dynamic_ncols=True,
                file=sys.stdout,
            )

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar.close()
            self.training_bar = None

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.is_local_process_zero:
            self.current_loss = float(metrics.get("train_loss"))

    def on_step_end(self, args, state, control, **kwargs):

        if state.is_local_process_zero:
            self.steps += 1
            avg_loss = 0
            desc = ""
            if (
                self.current_loss == self.current_loss
            ):  # don't add if current_loss is NaN
                avg_loss = self.average_loss(
                    self.current_loss, self.prev_avg_loss, self.smoothing
                )
                self.prev_avg_loss = avg_loss

            if self.current_loss:
                desc = f"Loss: {self.current_loss:.3f} — Avg: {avg_loss:.3f}"

            if state.global_step % self.refresh_rate == 0:
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
                self.training_bar.update(self.refresh_rate)
                if self.current_loss:
                    self.training_bar.set_description(desc)

            if self.save_every > 0 and self.steps % self.save_every == 0:
                self.save_pytorch_model()

            if self.generate_every > 0 and self.steps % self.generate_every == 0:
                self.generate_sample_text()

    def generate_sample_text(self):
        # only runs on state.is_local_process_zero
        self.training_bar.write(
            f"\033[1m{self.steps:,} steps reached: generating sample texts.\033[0m"
        )

        gen_length_max = getattr(self.model.config, "n_positions", None) or getattr(
            self.model.config, "max_position_embeddings", None
        )
        gen_length = min(gen_length_max, 256)

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None) or getattr(
            self.tokenizer, "eos_token_id", None
        )

        outputs = self.model.generate(
            input_ids=None,
            max_length=gen_length,
            do_sample=True,
            num_return_sequences=self.n_generate,
            temperature=0.7,
            pad_token_id=pad_token_id,
        )

        gen_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for text in gen_texts:
            self.training_bar.write("=" * 10)
            self.training_bar.write(text)

        self.training_bar.write("=" * 10)

    def save_pytorch_model(self):
        # only runs on state.is_local_process_zero
        self.training_bar.write(
            f"\033[1m{self.steps:,} steps reached: saving model to /{self.output_dir}\033[0m"
        )

        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        if self.save_gdrive:
            for pt_file in ["pytorch_model.bin", "config.json", "tokenizer.json"]:
                shutil.copyfile(
                    os.path.join(self.output_dir, pt_file),
                    os.path.join("/content/drive/My Drive/", self.run_id, pt_file),
                )

    def average_loss(self, current_loss, prev_avg_loss, smoothing):
        if prev_avg_loss is None:
            return current_loss
        else:
            return (smoothing * current_loss) + (1 - smoothing) * prev_avg_loss
