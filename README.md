# aitextgen

A robust tool for advanced AI text generation.

aitextgen is a Python package that leverages [PyTorch](https://pytorch.org), [Huggingface Transformers](https://github.com/huggingface/transformers) and [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) with specific optimizations for text generation, plus _many_ added features. It is the successor to textgenrnn and gpt-2-simple, merging the advantages of both packages.

- Finetunes on a pretrained GPT-2 model from OpenAI...or create your own GPT-2 model + tokenizer and train from scratch, even on your local computer without a GPU!
- Generates text faster than gpt-2-simple and with better memory efficiency, even moreso if you export your model to TorchScript!
- Model agnostic and future-proofed to support new developments in the Transformers-based world.
- With Transformers, aitextgen preserves compatibility with the base package, allowing you to use the model for other NLP tasks and upload to to the Huggingface model repository. Uses the `generate()` function to allow a massive amount of control over the generated text.
- With pytorch-lightning, aitextgen trains models not just on CPU and GPUs, but also _multiple_ GPUs and TPUs! Robust training progress support, with the ability to add optional loggers.
- The input dataset is its own object, allowing you to not only easily encode, cache, and compress them on a local computer before transporting it, but you are able to _merge_ datasets without biasing the resulting dataset, or _cross-train_ models so it learns some data fully and some partially to create blended output.

## Demo

You can use aitextgen _for free_ with powerful GPUs and TPUs using these Colaboratory Notebooks!

- Finetune an existing GPT-2 model (GPU; quick to set up)
- Finetune an existing GPT-2 model (TPUv2; _much_ faster training than GPU but limited to 124M/355M GPT-2 models and requires extra notebook setup)
- Train a GPT-2 model + tokenizer from scratch (TPUv2)

But if you just want to test aitextgen, you can train and generate a small model on your own computer with this Jupyter Notebook.

## Installation

aitextgen can be installed from PyPI:

```sh
pip3 install aitextgen
```

## Quick Example

Here's how you can quickly test out aitextgen on your own computer, even if you don't have a GPU!

For generating text from a pretrained GPT-2 model ([Jupyter Notebook](/notebooks/generation_hello_world.ipynb)):

```python
from aitextgen import aitextgen

# Without any parameters, aitextgen() will download, cache, and load the 124M GPT-2 "small" model
ai = aitextgen()

ai.generate()
ai.generate(n=3, max_length=100)
ai.generate(n=3, prompt="I believe in unicorns because", max_length=100)
ai.generate_to_file(n=10, prompt="I believe in unicorns because", max_length=100, temperature=1.2)
```

Want to train your own micro GPT-2 model on your own computer? Open up a Python console and go:

```python
import requests
import os
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import GPT2ConfigCPU
from aitextgen import aitextgen

# Download a Shakespeare text for training
file_name = "shakespeare.txt"
if not os.path.isfile(file_name):
	url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
	data = requests.get(url)

	with open(file_name, 'w') as f:
		f.write(data.text)

# Train a custom BPE Tokenizer on the downloaded text
# This will save two files: aitextgen-vocab.json and aitextgen-merges.txt,
# which are needed to rebuild the tokenizer.
train_tokenizer(file_name)

# GPT2ConfigCPU is a microvariant of GPT-2 optimized for CPU-training
config = GPT2ConfigCPU()

# Instantiate aitextgen using the created tokenizer and config
ai = aitextgen(vocab_file="aitextgen-vocab.json",
			   merges_file="aitextgen-merges.txt",
			   config=config)

# Train the model! It will save pytorch_model.bin after completion
ai.train(file_name, batch_size=16)

# Generate text from it!
ai.generate(5)
```

Want to run aitextgen and finetune GPT-2? Use the Colab notebooks in the Demos section, or follow the documentation to see how you can run these tools on Google Cloud Platform at maximum cost efficiency!

## Helpful Notes

- To convert a GPT-2 model trained using earlier TensorFlow-based finetuning tools such as gpt-2-simple to the PyTorch format, use the transformers-cli command and the [instructions here](https://huggingface.co/transformers/converting_tensorflow_models.html) to convert the checkpoint (where `OPENAI_GPT2_CHECKPOINT_PATH` is the _folder_ containing the model)
- When running on Google Cloud Platform (including Google Colab), it's recommended to download the TF-based GPT-2 from the Google API vs. downloading the PyTorch GPT-2 from Huggingface as the download will be _much_ faster and also saves Huggingface some bandwidth.
- If you want to generate text from a GPU, you must manually move the model to the GPU (it will not be done automatically to save GPU VRAM for training). Either call `to_gpu=True` when loading the model or call `to_gpu()` on the aitextgen object.

## Upcoming Features

The current release (v0.1) of aitextgen **is considered to be a beta**, targeting the most common use cases. The Notebooks and examples written so far are tested to work, but more fleshing out of the docs/use cases will be done over the next few months.

Immediate priorities are:

- fp16 fixes

The next versions of aitextgen (and one of the reasons I made this package in the first place) will have native support for _schema-based generation_. (see [this repo](https://github.com/minimaxir/gpt-2-keyword-generation) for a rough proof-of-concept)

Additionally, I plan to develop an aitextgen [SaaS](https://en.wikipedia.org/wiki/Software_as_a_service) to allow anyone to run aitextgen in the cloud and build APIs/Twitter+Slack+Discord bots with just a few clicks. (the primary constraint is compute cost; if any venture capitalists are interested in funding the development of such a service, let me know)

I've listed more tenative features in the [UPCOMING](UPCOMING.md) document.

## Ethics

aitextgen is a tool primarily intended to help facilitate creative content. It is not a tool intended to decive. Although parody accounts are an obvious use case for this package, make sure you are _as upfront as possible_ with the methodology of the text you create. This includes:

- State that the text was generated using aitextgen and/or a GPT-2 model architecture. (a link to this repo would be a bonus!)
- If parodying a person, explicitly state that it is a parody, and reference who it is parodying.
- If the generated human-curated, or if it's unsupervised random output
- Indicating who is maintaining/curating the AI-generated text.
- Make a good-faith effort to remove overfit output from the generated text that matches the input text verbatim.

It's fun to anthropomorphise the nameless "AI" as an absent genius, but part of the reason I made aitextgen (and all my previous text-generation projects) is to make the technology more accessible and accurately demonstrate both its promise, and its limitations. **Any AI text generation projects that are deliberately deceptive may be disavowed.**

## Maintainer/Creator

Max Woolf ([@minimaxir](https://minimaxir.com))

_Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir) and [GitHub Sponsors](https://github.com/sponsors/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use._

## License

MIT
