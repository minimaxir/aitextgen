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
import csv
import os
import re
import logging
from tqdm import trange
from datetime import datetime
from random import randint
from .TokenDataset import TokenDataset
from .utils import *

logger = logging.getLogger(__name__)


class aitextgen:
    def __init__(
        self, model=None, config=None, cache_dir="aitextgen", tf_gpt2=None
    ):

        if tf_gpt2 is not None:
            if model is None:
                assert tf_gpt2 in [
                    "124M",
                    "355M",
                    "774M",
                    "1558M",
                ], "Invalid GPT-2 model size."

                download_gpt2(cache_dir, tf_gpt2)

                gpt2_path = os.path.join(cache_dir, tf_gpt2)

                if not os.path.isfile(
                    os.path.join(gpt2_path, "pytorch_model.bin")
                ):
                    logger.info(
                        "Converting the GPT-2 TensorFlow weights to PyTorch."
                    )
                    convert_gpt2_checkpoint_to_pytorch(gpt2_path, "", gpt2_path)

                model = gpt2_path
            logger.info("Loading GPT-2 model from %s.", model)
            self.model = AutoModelWithLMHead.from_pretrained(
                model, config=GPT2Config(),
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "gpt2", cache_dir=cache_dir
            )

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
    ):

        if prompt:
            prompt_text = prompt
            prompt = encode_text(prompt, self.tokenizer)

        if not bos_token:
            bos_token_id = self.tokenizer.bos_token_id

        if not eos_token:
            eos_token_ids = self.tokenizer.eos_token_id

        outputs = self.model.generate(
            input_ids=prompt,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
            bos_token_id=bos_token_id,
            eos_token_ids=eos_token_ids,
            num_return_sequences=n,
        )

        if n > 1:
            gen_texts = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs[0]
            ]
        else:
            gen_texts = [
                self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            ]

        if not return_as_list:
            if prompt is not None:
                # Bold the prompt if printing to console
                gen_texts = [
                    re.sub(
                        r"^" + prompt_text,
                        "\033[1m" + prompt_text + "\033[0m",
                        text,
                    )
                    for text in gen_texts
                ]

            print(*gen_texts, sep="\n" + "=" * 10 + "\n")
        else:
            return gen_texts

    def generate_one(self, **kwargs):
        """
        Generates a single text, and returns it as a string.

        Useful for returning a generated text within an API.
        """

        return self.generate(n=1, return_as_list=True, **kwargs)[0]

    def generate_samples(self, n=3, temperatures=[0.7, 1.0, 1.2], **kwargs):
        """
        Prints multiple samples to console at specified temperatures.
        """

        for temperature in temperatures:
            print(
                "#" * 20 + "\nTemperature: {}\n".format(temperature) + "#" * 20
            )
            self.generate(
                n=n, temperature=temperature, return_as_list=False, **kwargs
            )

    def generate_to_file(
        self,
        n=20,
        batch_size=5,
        destination_path=None,
        sample_delim="=" * 20 + "\n",
        seed=None,
        **kwargs
    ):

        """
        Generates a bulk amount of texts to a file.
        """

        assert n % batch_size == 0, "n must be divisible by batch_size."

        if destination_path is None:
            # Create a time-based file name to prevent overwriting.
            # Use a 8-digit number as the seed, which is the last
            # numeric part of the file name.
            if seed is None:
                seed = randint(10000000, 100000000 - 1)
            assert isinstance(seed, int)
            destination_path = "aitextgen_{:%Y%m%d_%H%M%S}_{}.txt".format(
                datetime.utcnow(), seed
            )

        logging.info("Generating {:,} texts to {}".format(n, destination_path))

        pbar = trange(n)
        f = open(destination_path, "w", encoding="utf-8")

        for _ in range(n // batch_size - 1):
            gen_texts = self.generate(
                n=n, return_as_list=True, seed=seed, **kwargs
            )

            for gen_text in gen_texts:
                f.write("{}\n{}".format(gen_text, sample_delim))
            pbar.update(batch_size)

        pbar.close()
        f.close()


def encode_text(text, tokenizer):
    """
    Encodes text into an id-based tensor.
    """

    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)
