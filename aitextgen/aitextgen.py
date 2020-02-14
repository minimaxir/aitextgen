from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
from torch import tensor
from torch.utils.data import Dataset
import csv
import os
import re


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

    def generate(self, prompt=None, max_length=200,
                 temperature=1.0, do_sample=True,
                 bos_token=None,
                 eos_token=None,
                 return_as_list=False):

        if prompt:
            prompt_tokens = encode_text(prompt, self.tokenizer)

        if not bos_token:
            bos_token_id = self.tokenizer.bos_token_id

        if not eos_token:
            eos_token_ids = self.tokenizer.eos_token_id

        outputs = self.model.generate(
            input_ids=prompt_tokens,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
            bos_token_id=bos_token_id,
            eos_token_ids=eos_token_ids
        )

        gen_text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True)

        if not return_as_list:
            if prompt is not None:
                # Bold the prompt if printing to console
                gen_text = re.sub(r'^' + prompt,
                                  '\033[1m' + prompt + '\033[0m',
                                  gen_text)

            print(gen_text)
        else:
            return [gen_text]


def encode_text(text, tokenizer):
    """
    Encodes text into an id-based tensor.
    """

    return tensor(tokenizer.encode(
        text)).unsqueeze(0)
