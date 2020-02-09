from transformers import AutoModelWithLMHead, AutoTokenizer
import torch


class aitextgen:

    def __init__(self, model=None, config=None):

        if model == None:
            self.model = AutoModelWithLMHead.from_pretrained(
                'distilgpt2', cache_dir='')
            self.tokenizer = AutoTokenizer.from_pretrained(
                'distilgpt2', cache_dir='')

    def generate(self, prefix=None, max_length=200,
                 temperature=1.0, do_sample=True):

        if prefix:
            prefix = encode_text(prefix, self.tokenizer)

        outputs = self.model.generate(
            input_ids=prefix,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample
        )

        gen_text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True)

        print(gen_text)


def encode_text(text, tokenizer):
    """
    Encodes text into an id-based tensor.
    """

    return torch.tensor(tokenizer.encode(
        text)).unsqueeze(0)
