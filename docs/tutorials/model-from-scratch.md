# Training a GPT-2 Model From Scratch

The original GPT-2 model released by OpenAI was trained on English webpages linked to from Reddit, with a strong bias toward longform content (multiple paragraphs).

If that is _not_ your use case, you may get a better generation quality _and_ speed by training your own model and Tokenizer. Examples of good use cases:

- Short-form content (e.g. Tweets, Reddit post titles)
- Non-English Text
- Encoded Text

It still will require a _massive_ amount of training time (several hours, even on a TPU), but will be more flexible.

## Building a Custom Tokenizer.

The `train_tokenizer()` function from `aitextgen.tokenizers` trains the model on the specified text(s) on disk.

<!--prettier-ignore-->
!!! note "Vocabulary Size"
    The default vocabulary size for `train_tokenizer()` is 5,000 tokens. Although this is much lower than GPT-2's 50k vocab size, the smaller the vocab size, the easier it is to train the model (since it's more likely for the model to make a correct "guess"), and the model file size will be _much_ smaller.

```python
from aitextgen.tokenizers import train_tokenizer
train_tokenizer(file_name)
```

This creates two files: `aitextgen-vocab.json` and `aitextgen-merges.txt`, which are needed to rebuild the tokenizer.

# Building a Custom Dataset

You can build a TokenDataset based off your custom Tokenizer, to be fed into the model.

```python
data = TokenDataset(file_name, vocab_file=vocab_file, merges_file=merges_file, block_size=32)
```

## Building a Custom Config

Whenever you load a default 124M GPT-2 model, it uses a `GPT2Config()` under the hood. But you can create your own, with whatever parameters you want.

The `build_gpt2_config()` function from `aitextgen.utils` gives you more control.

```python
config = build_gpt2_config(vocab_size=5000, max_length=32, dropout=0.0, n_embd=256, n_layer=8, n_head=8)
```

A few notes on the inputs:

- `vocab_size`: Vocabulary size: this _must_ match what you used to build the tokenizer!
- `max_length`: Context window for the GPT-2 model: this _must_ match the `block_size` used in the TokenDataset!
- `dropout`: Dropout on various areas of the model to limit overfitting (you should likely keep at 0)
- `n_embd`: The embedding size for each vocab token.
- `n_layers`: Transformer layers
- `n_head`: Transformer heads

<!--prettier-ignore-->
!!! note "Model Size"
    GPT-2 Model size is directly proportional to `vocab_size` \* `embeddings`.

## Training the Custom Model

You can instantiate an empty GPT-2 according to your custom config, and construct a custom tokenizer according to your vocab and merges file:

```python
ai = aitextgen(vocab_file=vocab_file, merges_file=merges_file, config=config)
```

Training is done as normal.

```python
ai.train(data, batch_size=16, num_steps=5000)
```

## Reloading the Custom Model

You'll always need to provide the vocab_file, merges_file, and config (a config file is saved when the model is saved; you can either build it at runtime as above, or use the `config.json`)

```python
ai = aitextgen(model="pytorch_model.bin", vocab_file=vocab_file, merges_file=merges_file, config=config)
```
