# Hello World

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

Want to train your own mini GPT-2 model on your own computer? Download this [text file of Shakespeare plays](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt), cd to that directory in a Teriminal, open up a `python3` console and go:

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
