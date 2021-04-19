# aitextgen

_Last Updated: April 18th, 2021 (aitextgen v0.5.0)_

A robust Python tool for text-based AI training and generation using [OpenAI's](https://openai.com) [GPT-2](https://openai.com/blog/better-language-models/) and [EleutherAI's](https://www.eleuther.ai) [GPT Neo/GPT-3](https://github.com/EleutherAI/gpt-neo) architecture.

aitextgen is a Python package that leverages [PyTorch](https://pytorch.org), [Hugging Face Transformers](https://github.com/huggingface/transformers) and [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) with specific optimizations for text generation using GPT-2, plus _many_ added features. It is the successor to [textgenrnn](https://github.com/minimaxir/textgenrnn) and [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple), taking the best of both packages:

- Finetunes on a pretrained 124M/355M/774M GPT-2 model from OpenAI or a 125M/355M GPT Neo model from EleutherAI...or create your own GPT-2/GPT Neo model + tokenizer and train from scratch!
- Generates text faster than gpt-2-simple and with better memory efficiency!
- With Transformers, aitextgen preserves compatibility with the base package, allowing you to use the model for other NLP tasks, download custom GPT-2 models from the HuggingFace model repository, and upload your own models! Also, it uses the included `generate()` function to allow a massive amount of control over the generated text.
- With pytorch-lightning, aitextgen trains models not just on CPUs and GPUs, but also _multiple_ GPUs and (eventually) TPUs! It also includes a pretty training progress bar, with the ability to add optional loggers.
- The input dataset is its own object, allowing you to not only easily encode megabytes of data in seconds, cache, and compress it on a local computer before transporting to a remote server, but you are able to _merge_ datasets without biasing the resulting dataset, or _cross-train_ on multiple datasets to create blended output.

## Installation

aitextgen can be installed from PyPI:

```sh
pip3 install aitextgen
```

## More

For more info on how to use aitextgen, check out the nav sidebar, or you can do a [Hello World tutorial](tutorials/hello-world.md).
