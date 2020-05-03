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
    pad_token: str = "<<<pad>>>",
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

    tokenizer.add_special_tokens(
        {
            "bos_token": bos_token,
            "eos_token": eos_token,
            "unk_token": unk_token,
            "pad_token": pad_token,
        }
    )

    return tokenizer
