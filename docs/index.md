# aitextgen

A robust tool for advanced AI text generation via [GPT-2](https://openai.com/blog/better-language-models/).

aitextgen is a Python package that leverages [PyTorch](https://pytorch.org), [Huggingface Transformers](https://github.com/huggingface/transformers) and [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) with specific optimizations for text generation using GPT-2, plus _many_ added features. It is the successor to [textgenrnn](https://github.com/minimaxir/textgenrnn) and [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple), taking the best of both packages:

- Finetunes on a pretrained 124M GPT-2 model from OpenAI...or create your own GPT-2 model + tokenizer and train from scratch!
- Generates text faster than gpt-2-simple and with better memory efficiency! (even [from the 1.5B GPT-2 model](tutorials/generate_1_5b/)!)
- With Transformers, aitextgen preserves compatibility with the base package, allowing you to use the model for other NLP tasks, download custom GPT-2 models from the Huggingface model repository, and upload your own models! Also, it uses the included `generate()` function to allow a massive amount of control over the generated text.
- With pytorch-lightning, aitextgen trains models not just on CPUs and GPUs, but also _multiple_ GPUs and (eventually) TPUs! It also includes a pretty training progress bar, with the ability to add optional loggers.
- The input dataset is its own object, allowing you to not only easily encode megabytes of data in seconds, cache, and compress it on a local computer before transporting to a remote server, but you are able to _merge_ datasets without biasing the resulting dataset, or _cross-train_ on multiple datasets to create blended output.

## Installation

aitextgen can be installed from PyPI:

```sh
pip3 install aitextgen
```

## More

For more info on how to use aitextgen, check out the nav sidebar, or you can do a [Hello World tutorial](tutorials/hello-world.md).
