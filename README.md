# aitextgen

A robust Python tool for text-based AI training and generation using [OpenAI's](https://openai.com) [GPT-2](https://openai.com/blog/better-language-models/) architecture.

aitextgen is a Python package that leverages [PyTorch](https://pytorch.org), [Hugging Face Transformers](https://github.com/huggingface/transformers) and [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) with specific optimizations for text generation using GPT-2, plus _many_ added features. It is the successor to [textgenrnn](https://github.com/minimaxir/textgenrnn) and [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple), taking the best of both packages:

- Finetunes on a pretrained 124M GPT-2 model from OpenAI...or create your own GPT-2 model + tokenizer and train from scratch!
- Generates text faster than gpt-2-simple and with better memory efficiency! (even [from the 1.5B GPT-2 model](https://docs.aitextgen.io/tutorials/generate_1_5b/)!)
- With Transformers, aitextgen preserves compatibility with the base package, allowing you to use the model for other NLP tasks, download custom GPT-2 models from the Hugging Face model repository, and upload your own models! Also, it uses the included `generate()` function to allow a massive amount of control over the generated text.
- With pytorch-lightning, aitextgen trains models not just on CPUs and GPUs, but also _multiple_ GPUs and (eventually) TPUs! It also includes a pretty training progress bar, with the ability to add optional loggers.
- The input dataset is its own object, allowing you to not only easily encode megabytes of data in seconds, cache, and compress it on a local computer before transporting to a remote server, but you are able to _merge_ datasets without biasing the resulting dataset, or _cross-train_ on multiple datasets to create blended output.

You can read more about aitextgen [in the documentation](https://docs.aitextgen.io/)!

## Demo

You can play with aitextgen _for free_ with powerful GPUs using these Colaboratory Notebooks!

- [Finetune OpenAI's 124M GPT-2 model on your own dataset (GPU)](https://colab.research.google.com/drive/15qBZx5y9rdaQSyWpsreMDnTiZ5IlN0zD?usp=sharing)
- [Train a GPT-2 model + tokenizer from scratch (GPU)](https://colab.research.google.com/drive/144MdX5aLqrQ3-YW-po81CQMrD6kpgpYh?usp=sharing)

You can also play with custom [Reddit](notebooks/reddit_demo.ipynb) and [Hacker News](notebooks/hacker_news_demo.ipynb) demo models on your own PC.

## Installation

aitextgen can be installed [from PyPI](https://pypi.org/project/aitextgen/):

```sh
pip3 install aitextgen
```

## Quick Examples

Here's how you can quickly test out aitextgen on your own computer, even if you don't have a GPU!

For generating text from a pretrained GPT-2 model:

```python
from aitextgen import aitextgen

# Without any parameters, aitextgen() will download, cache, and load the 124M GPT-2 "small" model
ai = aitextgen()

ai.generate()
ai.generate(n=3, max_length=100)
ai.generate(n=3, prompt="I believe in unicorns because", max_length=100)
ai.generate_to_file(n=10, prompt="I believe in unicorns because", max_length=100, temperature=1.2)
```

You can also generate from the command line:

```sh
aitextgen generate
aitextgen generate --prompt "I believe in unicorns because" --to_file False
```

Want to train your own mini GPT-2 model on your own computer? Download this [text file of Shakespeare's plays](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt), cd to that directory in a Terminal, open up a `python3` console and go:

```python
from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import GPT2ConfigCPU
from aitextgen import aitextgen

# The name of the downloaded Shakespeare text for training
file_name = "input.txt"

# Train a custom BPE Tokenizer on the downloaded text
# This will save two files: aitextgen-vocab.json and aitextgen-merges.txt,
# which are needed to rebuild the tokenizer.
train_tokenizer(file_name)
vocab_file = "aitextgen-vocab.json"
merges_file = "aitextgen-merges.txt"

# GPT2ConfigCPU is a mini variant of GPT-2 optimized for CPU-training
# e.g. the # of input tokens here is 64 vs. 1024 for base GPT-2.
config = GPT2ConfigCPU()

# Instantiate aitextgen using the created tokenizer and config
ai = aitextgen(vocab_file=vocab_file, merges_file=merges_file, config=config)

# You can build datasets for training by creating TokenDatasets,
# which automatically processes the dataset with the appropriate size.
data = TokenDataset(file_name, vocab_file=vocab_file, merges_file=merges_file, block_size=64)

# Train the model! It will save pytorch_model.bin periodically and after completion.
# On a 2016 MacBook Pro, this took ~25 minutes to run.
ai.train(data, batch_size=16, num_steps=5000)

# Generate text from it!
ai.generate(10, prompt="ROMEO:")
```

Want to run aitextgen and finetune GPT-2? Use the Colab notebooks in the Demos section, or [follow the documentation](https://docs.aitextgen.io/) to get more information and learn some helpful tips!

## Known Issues

- TPUs cannot be used to train a model: although you _can_ train an aitextgen model on TPUs by setting `n_tpu_cores=8` in an appropriate runtime, and the training loss indeed does decrease, there are a number of miscellaneous blocking problems. [[Tracking GitHub Issue](https://github.com/minimaxir/aitextgen/issues/3)]
- TorchScript exporting, although it works with `ai.export()`, behaves oddly when reloaded back into Python, and is therefore not supported (yet). [[Tracking GitHub Issue](https://github.com/minimaxir/aitextgen/issues/5)]
- Finetuning the 355M GPT-2 model or larger on a GPU will cause the GPU to go OOM, even 16 GB VRAM GPUs (355M _does_ work with FP16 + 16 GB VRAM however). This is a [known issue with the Transformers GPT-2 implementation](https://github.com/huggingface/transformers/pull/2356), unfortunately. Gradient checkpointing may need to be implemented within the training loop of aitextgen. [[Tracking GitHub Issue](https://github.com/minimaxir/aitextgen/issues/6)]
  - As a temporary workaround, you can finetune larger models with [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple), then [convert the TensorFlow weights to a PyTorch model](https://docs.aitextgen.io/gpt-2-simple/).

## Upcoming Features

The current release (v0.2.X) of aitextgen **is considered to be a beta**, targeting the most common use cases. The Notebooks and examples written so far are tested to work, but more fleshing out of the docs/use cases will be done over the next few months in addition to fixing the known issues noted above.

The next versions of aitextgen (and one of the reasons I made this package in the first place) will have native support for _schema-based generation_. (See [this repo](https://github.com/minimaxir/gpt-2-keyword-generation) for a rough proof-of-concept.)

Additionally, I plan to develop an aitextgen [SaaS](https://en.wikipedia.org/wiki/Software_as_a_service) to allow anyone to run aitextgen in the cloud and build APIs/Twitter+Slack+Discord bots with just a few clicks. (The primary constraint is compute cost; if any venture capitalists are interested in funding the development of such a service, let me know.)

I've listed more tentative features in the [UPCOMING](UPCOMING.md) document.

## Ethics

aitextgen is a tool primarily intended to help facilitate creative content. It is not a tool intended to deceive. Although parody accounts are an obvious use case for this package, make sure you are _as upfront as possible_ with the methodology of the text you create. This includes:

- State that the text was generated using aitextgen and/or a GPT-2 model architecture. (A link to this repo would be a bonus!)
- If parodying a person, explicitly state that it is a parody, and reference who it is parodying.
- If the generated text is human-curated, or if it's unsupervised random output.
- Indicating who is maintaining/curating the AI-generated text.
- Make a good-faith effort to remove overfit output from the generated text that matches the input text verbatim.

It's fun to anthropomorphise the nameless "AI" as an abstract genius, but part of the reason I made aitextgen (and all my previous text-generation projects) is to make the technology more accessible and accurately demonstrate both its promise, and its limitations. **Any AI text generation projects that are deliberately deceptive may be disavowed.**

## Maintainer/Creator

Max Woolf ([@minimaxir](https://minimaxir.com))

_Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir) and [GitHub Sponsors](https://github.com/sponsors/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use._

## License

MIT
