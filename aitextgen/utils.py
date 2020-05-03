import os
import requests
from tqdm.auto import tqdm
import torch
import numpy as np
import random
from transformers import PretrainedConfig, GPT2Config


def download_gpt2(model_dir: str = "tf_model", model_name: str = "124M") -> None:
    """
    Downloads the GPT-2 model (weights only) into the specified directory
    from Google Cloud Storage.

    If running in Colaboratory or Google Compute Engine,
    this is substantially faster (and cheaper for HuggingFace) than using the
    default model downloading. However, the model is in TensorFlow,
    so the weights must be converted.

    Adapted from gpt-2-simple.
    """

    # create the <model_dir>/<model_name> subdirectory if not present
    sub_dir = os.path.join(model_dir, model_name)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    sub_dir = sub_dir.replace("\\", "/")  # needed for Windows

    for file_name in [
        "checkpoint",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
    ]:
        if not os.path.isfile(os.path.join(sub_dir, file_name)):
            download_file_with_progress(
                url_base="https://storage.googleapis.com/gpt-2",
                sub_dir=sub_dir,
                model_name=model_name,
                file_name=file_name,
            )


def download_file_with_progress(
    url_base: str, sub_dir: str, model_name: str, file_name: str
):
    """
    General utility for incrementally downloading files from the internet
    with progress bar.

    Adapted from gpt-2-simple.
    """

    # set to download 1MB at a time. This could be much larger with no issue
    DOWNLOAD_CHUNK_SIZE = 1024 * 1024
    r = requests.get(
        os.path.join(url_base, "models", model_name, file_name), stream=True
    )
    with open(os.path.join(sub_dir, file_name), "wb") as f:
        file_size = int(r.headers["content-length"])
        with tqdm(
            ncols=100, desc="Fetching " + file_name, total=file_size, unit_scale=True,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)
                pbar.update(DOWNLOAD_CHUNK_SIZE)


def encode_text(text: str, tokenizer):
    """
    Encodes text into an id-based tensor using the given tokenizer.
    """

    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)


def set_seed(seed: int):
    """
    Sets the seed for all potential generation libraries.
    """

    assert isinstance(seed, int), "seed must be an integer."
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_seed():
    """
    Resets the seed for all potential generation libraries.
    """
    random.seed()
    np.random.seed()
    torch.seed()
    torch.cuda.seed_all()


def build_config(changes: dict, cache_dir: str, base_config: str = "gpt2"):
    """
    Builds a custom config based on a given Transformers config,
    with a few more user-friendly aliases.
    """

    # Download but don't cache yet
    config = PretrainedConfig.from_pretrained(
        base_config, cache_dir=cache_dir
    ).to_dict()

    # use max_length as an alias for context window
    if "max_length" in changes:
        for key in ["n_positions", "n_ctx"]:
            changes[key] = changes["max_length"]

    # use dropout for relevant dropouts during training only
    if "dropout" in changes:
        for key in ["resid_pdrop", "embd_pdrop", "attn_pdrop"]:
            changes[key] = changes["dropout"]

    config.update(changes)
    new_config = PretrainedConfig.from_dict(config)

    return new_config


def GPT2ConfigCPU(**kwargs):
    """
    Returns a GPT-2 config more suitable for training on a regular consumer CPU.
    """

    return GPT2Config(
        n_positions=64, n_ctx=64, n_embd=128, n_layer=3, n_head=3, **kwargs
    )
