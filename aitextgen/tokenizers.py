from tokenizers import ByteLevelBPETokenizer
from typing import Union, List
import logging

logger = logging.getLogger(__name__)


def train_tokenizer(
    files: Union[str, List[str]],
    dropout: float = None,
    vocab_size: int = 5000,
    min_frequency: int = 2,
    save_path: str = "",
    added_tokens: List[str] = [],
    bos_token: str = "<|endoftext|>",
    eos_token: str = "<|endoftext|>",
    unk_token: str = "<|endoftext|>",
) -> None:
    """
    Tokenizes the text(s) as a tokenizer, wrapping the tokenizer package.
    See: https://huggingface.co/blog/how-to-train

    For consistency, this function makes opinionated assuptions.

    :param files: path to file(s) to train tokenizer on
    :param dropout: Trainign dropout
    :param vocab_size: Final vocabulary size
    :param min_frequency: Minimum number of occurences to add to vocab
    :param save_path: Where to save the final tokenizer
    :param added_tokens: List of tokens to add to the tokenizer (currently not working)
    :param bos_token: Beginning-of-string special token
    :param eos_token: End-of-string special token
    :param unk_token: Unknown special token
    """

    assert isinstance(files, str) or isinstance(
        files, list
    ), "files must be a string or a list."

    tokenizer = ByteLevelBPETokenizer(dropout=dropout)

    tokenizer.train(
        files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[bos_token, eos_token, unk_token],
        show_progress=True,
    )

    # Currently doesn't do anything
    # See: https://github.com/huggingface/tokenizers/issues/233
    # tokenizer.add_tokens(added_tokens)

    PREFIX = "aitextgen"
    save_path_str = "the current directory" if save_path == "" else save_path
    logger.info(
        f"Saving {PREFIX}-vocab.json and {PREFIX}-merges.txt to {save_path_str}. "
        + "You will need both files to build the GPT2Tokenizer."
    )
    tokenizer.save(save_path, PREFIX)
