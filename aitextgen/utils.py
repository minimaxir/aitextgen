import os
import requests
from tqdm.auto import tqdm


def download_gpt2(model_dir="tf_model", model_name="124M"):
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


def download_file_with_progress(url_base, sub_dir, model_name, file_name):
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
