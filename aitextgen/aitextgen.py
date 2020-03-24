from transformers import (
    AutoModelWithLMHead,
    AutoTokenizer,
    GPT2Config,
    AutoModelForPreTraining,
)
from transformers.convert_gpt2_original_tf_checkpoint_to_pytorch import (
    convert_gpt2_checkpoint_to_pytorch,
)
import torch
from torch.utils.data import Dataset
import time
import os
import re
import logging
from tqdm import trange
from datetime import datetime
from random import randint
from .TokenDataset import TokenDataset
import pytorch_lightning as pl
import random
import numpy as np
from .utils import *
from .train import *

logger = logging.getLogger(__name__)


class aitextgen:
    """
    Class that serves as the main aitextgen object for training and generation.

    ## Parameters

    * **model**: transformers model. If None, uses distilgpt2.
    * **config**: transformers config for the model. If None, uses distilgpt2.
    * **cache_dir**: folder path which has the current model alredy
    * tf_gpt2: folder path to the OpenAI-distributed version of GPT-2. This
    will convert the model to PyTorch if not present.
    """

    def __init__(self, model=None, config=None, cache_dir="aitextgen", tf_gpt2=None):

        if tf_gpt2 is not None:
            if model is None:
                assert tf_gpt2 in [
                    "124M",
                    "355M",
                    "774M",
                    "1558M",
                ], "Invalid GPT-2 model size."

                download_gpt2(cache_dir, tf_gpt2)

                if not os.path.isfile(os.path.join(cache_dir, "pytorch_model.bin")):
                    logger.info("Converting the GPT-2 TensorFlow weights to PyTorch.")
                    convert_gpt2_checkpoint_to_pytorch(
                        os.path.join(cache_dir, tf_gpt2), "", cache_dir
                    )

                model = os.path.join(cache_dir, "pytorch_model.bin")
            logger.info(f"Loading GPT-2 model from {model}.")
            self.model = AutoModelWithLMHead.from_pretrained(
                model, config=GPT2Config(),
            )
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)

        elif model is None:
            if len(os.listdir(cache_dir)) > 0:
                logger.info("Loading model from cache.")
            else:
                logger.info("Downloading model.")
            self.model = AutoModelWithLMHead.from_pretrained(
                "distilgpt2", cache_dir=cache_dir
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "distilgpt2", cache_dir=cache_dir
            )

    def generate(
        self,
        n=1,
        prompt=None,
        max_length=200,
        temperature=1.0,
        do_sample=True,
        bos_token=None,
        eos_token=None,
        return_as_list=False,
        seed=None,
    ):
        """
        Generates texts using the stored Transformers model.
        Currently generates text using the model's generate() function.

        ## Parameters

        * **n**: Numbers of texts to generate.
        * **prompt**: Text to force the generated text to start with
        * **max_length**: Maximum length for the generated text
        * **temperature**: Determines the "creativity" of the generated text.
        The value range is different for each type of Transformer.
        * **do_sample**: Samples the text, which is what we want. If False,
        the generated text will be the optimal prediction at each time,
        and therefore deterministic.
        * **bos_token**: Token which indicates the start of a text.
        Uses model setting if not set.
        * **bos_token**: Token which indicates the end of a text.
        Uses model setting if not set.
        * **return_as_list**: Boolean which determine if text should be returned
        as a list. If False, the generated texts will be print to console.
        * **seed**: A numeric seed which sets all randomness, allowing the
        generate text to be reprodible if rerunning with same parameters
        and model.
        """

        if prompt:
            prompt_text = prompt
            prompt = encode_text(prompt, self.tokenizer)

        if not bos_token:
            bos_token_id = self.tokenizer.bos_token_id

        if not eos_token:
            eos_token_ids = self.tokenizer.eos_token_id

        if seed:
            set_seed(seed)

        outputs = self.model.generate(
            input_ids=prompt,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
            bos_token_id=bos_token_id,
            eos_token_ids=eos_token_ids,
            num_return_sequences=n,
        )

        # Reset seed if used
        if seed:
            reset_seed()

        if n > 1:
            gen_texts = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs[0]
            ]
        else:
            gen_texts = [self.tokenizer.decode(outputs[0], skip_special_tokens=True)]

        if not return_as_list:
            if prompt is not None:
                # Bold the prompt if printing to console
                gen_texts = [
                    re.sub(
                        r"^" + prompt_text, "\033[1m" + prompt_text + "\033[0m", text,
                    )
                    for text in gen_texts
                ]

            print(*gen_texts, sep="\n" + "=" * 10 + "\n")
        else:
            return gen_texts

    def generate_one(self, **kwargs):
        """
        Generates a single text, and returns it as a string. Useful for
        returning a generated text within an API.

        See generate() for more parameters.
        """

        return self.generate(n=1, return_as_list=True, **kwargs)[0]

    def generate_samples(self, n=3, temperatures=[0.7, 1.0, 1.2], **kwargs):
        """
        Prints multiple samples to console at specified temperatures.
        """

        for temperature in temperatures:
            print("#" * 20 + f"\nTemperature: {temperature}\n" + "#" * 20)
            self.generate(n=n, temperature=temperature, return_as_list=False, **kwargs)

    def generate_to_file(
        self,
        n=20,
        batch_size=5,
        destination_path=None,
        sample_delim="=" * 20 + "\n",
        seed=None,
        **kwargs,
    ):

        """
        Generates a bulk amount of texts to a file, into a format
        good for manually inspecting and curating the texts.

        ## Parameters

        * **n**: Number of texts to generate
        * **batch_size**: Number of texts to generate simultaneously, taking
        advantage of CPU/GPU parallelization.
        * **destination_path**: File name of the file. If None, a timestampped
        file name is automatically used.
        * **sample_delim**: The text used to delimit each generated text.
        * **seed**: Seed used for the generation.

        See generate() for more parameters.
        """

        assert n % batch_size == 0, "n must be divisible by batch_size."

        if destination_path is None:
            # Create a time-based file name to prevent overwriting.
            # Use a 8-digit number as the seed, which is the last
            # numeric part of the file name.
            if seed is None:
                seed = randint(10 ** 7, 10 ** 8 - 1)

            destination_path = f"ATG_{datetime.utcnow():%Y%m%d_%H%M%S}_{seed}.txt"

        if seed:
            set_seed(seed)

        logging.info(f"Generating {n:,} texts to {destination_path}")

        pbar = trange(n)
        f = open(destination_path, "w", encoding="utf-8")

        for _ in range(n // batch_size - 1):
            gen_texts = self.generate(n=n, return_as_list=True, **kwargs)

            for gen_text in gen_texts:
                f.write("{}\n{}".format(gen_text, sample_delim))
            pbar.update(batch_size)

        pbar.close()
        f.close()

        if seed():
            reset_seed()

    def train(
        self,
        dataset=None,
        file_path=None,
        output_dir="",
        fp16=False,
        fp16_opt_level="O1",
        n_gpu=-1,
        n_tpu_cores=0,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        seed=None,
        learning_rate=5e-3,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        num_steps=5000,
        optimizer=None,
        loggers=None,
    ):
        """
        Trains/finetunes the model on the provided file/dataset using pytorch-lightning.

        ## Parameters

        * **dataset**: A TokenDataset containing the samples to be trained.
        * **file_path**: A string containing the text to be trained (shortcut
        instead of dataset)
        * **output_dir**: A string indicating where to store the resulting
        model file folder.
        * **fp16**: Boolean whether to use fp16, assuming using a compatible GPU/TPU.
        * **fp16_opt_level**: Option level for FP16/APEX training.
        * **n_gpu**: Number of GPU to use (-1 implies all available GPUs)
        * **n_tpu_cores**: Number of TPU cores to use (should be a multiple of 8)
        * **max_grad_norm**: Maximum gradient normalization
        * **gradient_accumulation_steps**: Number of gradient acc steps; can be increased
        to avoid going out-of-memory
        * **seed**: Interger representing the training seed.
        * **learning_rate**: Training learnign rate for the default AdamW optimizer.
        * **weight_decay**: Weight decay for the default AdamW optimizer.
        * **warmup_steps**: Warmrup steps for the default AdamW optimizer.
        * **num_steps**: Number of samples through the dataset.
        * **optimizer**: A PyTorch optimizer for the training, which will
        override the default AdamW optimizer + parameter settings.
        * **callbacks**: pytorch-lightning callbacks.
        * **loggers**: pytorch-lightning logger(s) to log results.
        """

        assert any(
            [dataset, file_path]
        ), "Either dataset or file_path must be specified"

        if file_path:
            dataset = TokenDataset(
                tokenizer=self.tokenizer, file_path=file_path, **kwargs
            )

        # Wrap the model in a pytorch-lightning module
        train_model = ATGTransformer(self.model, dataset, hparams)

        # Begin training
        if seed:
            set_seed(seed)

        if os.path.exists(output_dir) and os.listdir(output_dir):
            raise ValueError(
                "Output directory ({}) already exists and is not empty.".format(
                    output_dir
                )
            )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=output_dir,
            prefix="checkpoint",
            monitor="train_loss",
            mode="min",
            save_top_k=1,
        )

        train_params = dict(
            accumulate_grad_batches=gradient_accumulation_steps,
            gpus=n_gpu,
            max_steps=max_steps,
            show_progress_bar=True,
            gradient_clip_val=max_grad_norm,
            checkpoint_callback=checkpoint_callback,
            check_val_every_n_epoch=0,
        )

        if fp16:
            train_params["precision"] = 16 if fp16 else 32
            train_params["amp_level"] = fp16_opt_level

        if n_tpu_cores > 0:
            train_params["num_tpu_cores"] = n_tpu_cores
            train_params["gpus"] = 0

        # benchmark gives a boost for GPUs if input size is constant
        if n_gpu > 0:
            train_params["benchmark"] = True

        if n_gpu > 1:
            train_params["distributed_backend"] = "ddp"

        trainer = pl.Trainer(**train_params)
        trainer.fit(train_model)

        self.model = trainer.model

        if seed:
            reset_seed()


def encode_text(text, tokenizer):
    """
    Encodes text into an id-based tensor using the given tokenizer.
    """

    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)


def set_seed(seed):
    """
    Sets the seed for all potential generation libraries.
    """

    assert isinstance(seed, int), "seed must be an integer."
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_seed():
    """
    Resets the seed for all potential generation libraries.
    """
    random.seed()
    np.random.seed()
    torch.seed()
    torch.cuda.seed_all()
