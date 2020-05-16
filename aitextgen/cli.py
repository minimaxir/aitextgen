from .aitextgen import aitextgen
from .TokenDataset import TokenDataset
import fire


def aitextgen_cli(**kwargs):
    """Entrypoint for the CLI"""
    fire.Fire({"encode": encode_cli, "train": train_cli, "generate": generate_cli})


def encode_cli(file_path: str, **kwargs):
    """Encode + compress a dataset"""
    TokenDataset(file_path, save_cache=True, **kwargs)


def train_cli(file_path: str, **kwargs):
    """Train on a dataset."""
    ai = aitextgen(**kwargs)

    from_cache = file_path.endswith(".tar.gz")
    dataset = TokenDataset(file_path, from_cache=from_cache, **kwargs)

    ai.train(dataset, **kwargs)


def generate_cli(to_file: bool = True, **kwargs):
    """Generate from a trained model, or download one if not present."""

    ai = aitextgen(**kwargs)
    if to_file:
        ai.generate_to_file(**kwargs)
    else:
        ai.generate(**kwargs)
