# TokenDataset

aitextgen has a special class, `TokenDataset`, used for managing tokenized datasets to be fed into model training. (this is in contrast with other GPT-2 finetuning approaches, which tokenizes at training time although you can still do that by passing a `file_path` and other relevant parameters to `ai.train()`.)

This has a few nice bonuses, including:

- Tokenize a dataset on a local machine ahead of time and compress it, saving time/bandwidth transporting data to a remote machine
- Supports both reading a dataset line-by-line (including single-column CSVs), or bulk texts.
- Debug and log the loaded texts.
- Merge datasets together without using external libraries
- Cross-train on multiple datasets to "blend" them together.

## Creating a TokenDataset For GPT-2 Finetuning

The easiest way to create a TokenDataset is to provide a target file. If no `tokenizer_file` is provided, it will use the default GPT-2 tokenizer.

```py3
from aitextgen.TokenDataset import TokenDataset

data = TokenDataset("shakespeare.txt")
```

If you pass a single-column CSV and specify `line_by_line=True`, the TokenDataset will parse it row-by-row, and is the recommended way to handle multiline texts.

```py3
data = TokenDataset("politics.csv", line_by_line=True)
```

You can also manually pass a list of texts to `texts` instead if you've processed them elsewhere.

```py3
data = TokenDataset(texts = ["Lorem", "Ipsum", "Dolor"])
```

## Block Size

`block_size` is another parameter that can be passed when creating a TokenDataset, more useful for custom models. This should match the context window (e.g. the `n_positions` or `max_position_embeddings` config parameters). By default, it will choose `1024`: the GPT-2 context window.

When implicitly loading a dataset via `ai.train()`, the `block_size` will be set to what is supported by the corresponding model `config`.

## Debugging a TokenDataset

When loading a dataset, a progress bar will appear showing how many texts are loaded and

If you want to see what exactly is input to the model during training, you can access a slice via `data[0]`.

## Saving/Loading a TokenDataset

When creating a TokenDataset, you can automatically save it as a compressed gzipped numpy array when completed.

```py3
data = TokenDataset("shakespeare.txt", save_cache=True)
```

Or save it after you've loaded it with the `save()` function.

```py3
data = TokenDataset("shakespeare.txt")
data.save()
```

By default, it will save to `dataset_cache.tar.gz`. You can then reload that into another Python session by specifying the cache.

```py3
data = TokenDataset("dataset_cache.tar.gz", from_cache=True)
```

<!--prettier-ignore-->
!!! note "CLI"
    You can quickly create a Tokenized dataset using the command line, e.g. `aitextgen encode text.txt`. This will drastically reduce the file size, and is recommended before moving the file to cloud services (where it can be loaded using the `from_cache` parameter noted above)

## Using TokenDatasets with a Custom GPT-2 Model

The default TokenDataset has a `block_size` of `1024`, which corresponds to the _context window of the default GPT-2 model_. If you're using a custom model w/ a different maximum. Additionally, you must explicitly provide the tokenizer file to rebuild the tokenizer, as the tokenizer will be different than the normal GPT-2 one.

See the [Model From Scratch](tutorials/model-from-scratch.md) docs for more info.

## Merging TokenDatasets

Merging processed TokenDatasets can be done with the `merge_datasets()` function. By default, it will take samples equal to the smallest dataset from each TokenDataset, randomly sampling the appropriate number of texts from the larger datasets. This will ensure that model training does not bias toward one particular dataset. (it can be disabled by setting `equalize=False`)

(You _can_ merge bulk datasets and line-by-line datasets, but the training output may be bizarre!)

<!--prettier-ignore-->
!!! note "About Merging"
    The current implementation merges by subset count, so equalization may not be perfect, but it will not significantly impact training.

```py3
from aitextgen.TokenDataset import TokenDataset, merge_datasets

data1 = TokenDataset("politics1000.csv", line_by_line=True)   # 1000 samples
data2 = TokenDataset("politics4000.csv", line_by_line=True)   # 4000 samples

data_merged = merge_datasets([data1, data2])   # ~2000 samples
```
