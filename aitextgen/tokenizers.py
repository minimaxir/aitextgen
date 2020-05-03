from tokenizers import ByteLevelBPETokenizer
from typing import Union, List


def train_tokenizer(
    files: Union[str, List[str]],
    dropout: float = None,
    vocab_size: int = 10000,
    min_frequency: int = 2,
    bos_token: str = "<<<start>>>",
    eos_token: str = "<<<end>>>",
    unk_token: str = "<<<unk>>>",
):
    """
    Tokenizes the text(s) as a tokenizer, wrapping the tokenizer package.

    For consistency, this function makes opinionated assuptions.
    """

    assert isinstance(files, str) or isinstance(
        files, list
    ), "files must be a string or a list."

    tokenizer = ByteLevelBPETokenizer(dropout=dropout)
    tokenizer.train(files, vocab_size=vocab_size, min_frequency=min_frequency)

    tokenizer.bos_token = bos_token
    tokenizer.eos_token = eos_token
    tokenizer.unk_token - unk_token

    return tokenizer
