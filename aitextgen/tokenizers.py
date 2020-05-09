from tokenizers import ByteLevelBPETokenizer
from typing import Union, List
import logging

logger = logging.getLogger(__name__)


def train_tokenizer(
    files: Union[str, List[str]],
    dropout: float = None,
    vocab_size: int = 10000,
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
    """

    assert isinstance(files, str) or isinstance(
        files, list
    ), "files must be a string or a list."

    tokenizer = ByteLevelBPETokenizer(dropout=dropout)
    tokenizer.add_tokens(added_tokens)

    tokenizer.train(
        files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[bos_token, eos_token, unk_token],
        show_progress=True,
    )

    PREFIX = "aitextgen"
    save_path_str = "the current directory" if save_path == "" else save_path
    logger.info(
        f"Saving {PREFIX}-vocab.json and {PREFIX}-merges.txt to {save_path_str}. "
        + "You will need both files to build the GPT2Tokenizer."
    )
    tokenizer.save(save_path, PREFIX)
