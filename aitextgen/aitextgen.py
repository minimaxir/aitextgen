from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
from torch import tensor
from torch.utils.data import Dataset
import csv
import os


class aitextgen:

    def __init__(self, model=None, config=None, cache_dir='distilgpt2'):

        if model is None:
            if len(os.listdir(cache_dir)) > 0:
                print("Loading model from cache.")
            else:
                print("Downloading model.")
                self.model = AutoModelWithLMHead.from_pretrained(
                    'distilgpt2', cache_dir=cache_dir)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    'distilgpt2', cache_dir=cache_dir)

    def generate(self, prefix=None, max_length=200,
                 temperature=1.0, do_sample=True,
                 bos_token=None,
                 eos_token=None):

        if prefix:
            prefix = encode_text(prefix, self.tokenizer)

        if not bos_token:
            bos_token_id = self.tokenizer.bos_token_id

        if not eos_token:
            eos_token_ids = self.tokenizer.eos_token_id

        outputs = self.model.generate(
            input_ids=prefix,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
            bos_token_id=bos_token_id,
            eos_token_ids=eos_token_ids
        )

        gen_text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True)

        print(gen_text)


def encode_text(text, tokenizer):
    """
    Encodes text into an id-based tensor.
    """

    return tensor(tokenizer.encode(
        text)).unsqueeze(0)
