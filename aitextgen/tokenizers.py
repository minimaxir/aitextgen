from tokenizers import ByteLevelBPETokenizer
from typing import Union, List
from transformers import GPT2Tokenizer
import logging
import os
import json

logger = logging.getLogger(__name__)


def train_tokenizer(
    files: Union[str, List[str]],
    dropout: float = None,
    vocab_size: int = 10000,
    min_frequency: int = 2,
    save_path: str = "aitextgen",
    bos_token: str = "<<<start>>>",
    eos_token: str = "<<<end>>>",
    unk_token: str = "<<<unk>>>",
):
    """
    Tokenizes the text(s) as a tokenizer, wrapping the tokenizer package.
    See: https://huggingface.co/blog/how-to-train

    For consistency, this function makes opinionated assuptions.
    """

    assert isinstance(files, str) or isinstance(
        files, list
    ), "files must be a string or a list."

    tokenizer = ByteLevelBPETokenizer(dropout=dropout)
    tokenizer.add_special_tokens([bos_token, eos_token, unk_token])
    tokenizer.train(
        files, vocab_size=vocab_size, min_frequency=min_frequency, show_progress=True
    )

    # The saved files must have "gpt2" in its name.
    PREFIX = "gpt2-aitextgen"
    tokenizer.save(save_path, PREFIX)

    # Reload the generated vocab + merge file into a GPT2Tokenizer,
    # then resave so the Tokenizer is in the correct format
    tokenizer_gpt2 = GPT2Tokenizer(
        os.path.join(save_path, f"{PREFIX}-vocab.json"),
        os.path.join(save_path, f"{PREFIX}-merges.txt"),
        bos_token=bos_token,
        eos_token=eos_token,
        unk_token=unk_token,
    )

    tokenizer_gpt2.save_pretrained(save_path)

    with open("config.json", "w"):
        json.dumps({"model_type": "gpt2"})

    return tokenizer_gpt2
