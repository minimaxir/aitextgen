from tokenizers import ByteLevelBPETokenizer
from typing import Union, List


def train_tokenizer(
    files: Union[str, List[str]],
    dropout: float = None,
    vocab_size: int = 1000,
    min_frequency: int = 2,
    prefix: str = "aitextgen",
    save_path: str = "",
    added_tokens: List[str] = [],
    bos_token: str = "<|endoftext|>",
    eos_token: str = "<|endoftext|>",
    unk_token: str = "<|endoftext|>",
    serialize: bool = True,
    trim_offsets: bool = True,
) -> None:
    """
    Tokenizes the text(s) as a tokenizer, wrapping the tokenizer package.
    See: https://huggingface.co/blog/how-to-train

    For consistency, this function makes opinionated assuptions.

    :param files: path to file(s) to train tokenizer on
    :param dropout: Training dropout
    :param vocab_size: Final vocabulary size
    :param min_frequency: Minimum number of occurences to add to vocab
    :param prefix: File name prefix of the final tokenizer
    :param save_path: Where to save the final tokenizer
    :param added_tokens: List of tokens to add to the tokenizer (currently not working)
    :param bos_token: Beginning-of-string special token
    :param eos_token: End-of-string special token
    :param unk_token: Unknown special token
    """

    assert isinstance(files, str) or isinstance(
        files, list
    ), "files must be a string or a list."

    assert isinstance(added_tokens, list), "added_tokens must be a list."

    if isinstance(files, str):
        files = [files]

    tokenizer = ByteLevelBPETokenizer(dropout=dropout, trim_offsets=trim_offsets)

    tokenizer.train(
        files=files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[bos_token, eos_token, unk_token] + added_tokens,
    )

    if serialize:
        tokenizer.save(f"{prefix}.tokenizer.json")
    else:
        tokenizer.save_model(save_path, prefix)
