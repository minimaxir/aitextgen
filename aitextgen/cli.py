from .aitextgen import aitextgen
from .TokenDataset import TokenDataset
import fire


def aitextgen_cli(**kwargs):
    """Entrypoint for the CLI"""
    fire.fire()


def encode_cli(file_path: str, **kwargs):
    """Encode + compress a dataset"""
    TokenDataset(file_path, save_cache=True, **kwargs)


def train_cli(file_path: str, **kwargs):
    """Train on a dataset. Uses 124M gpt-2 by default"""
    ai = aitextgen(model="gpt2")

    from_cache = file_path.endswith(".tar.gz")
    dataset = TokenDataset(file_path, from_cache=from_cache, **kwargs)

    ai.train(dataset, **kwargs)


def generate(cache_dir: str, **kwargs):
    """Generate from a trained model"""

    ai = aitextgen(cache_dir=cache_dir)
    ai.generate_to_file(**kwargs)
