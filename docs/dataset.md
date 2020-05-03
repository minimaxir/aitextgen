# TokenDataset

aitextgen has a special class, `TokenDataset`, used for managing tokenized datasets to be fed into model training. (this is in contrast with other GPT-2 finetuning approaches, which tokenizes at training time although you can still do that if you want)

This has a few nice bonuses, including:

- Tokenize a dataset on a local machine ahead of time and compress it, saving time/bandwidth transporting data to a remote machine
- Supports both reading a dataset line-by-line (including single-column CSVs), or bulk texts.
- Merge datasets together without using external libraries
- Cross-train on multiple datasets to "blend" them together.

## Creating a TokenDataset For GPT-2 Finetuning

The easiest way to create a TokenDataset is to provide a target file. If no value to `tokenizer` is provided, it will use the default GPT-2 tokenizer.

```python
from aitextgen.TokenDataset import TokenDataset

data = TokenDataset("shakespeare.txt")
```

If you pass a single-column CSV, the TokenDataset will parse it row-by-row, and is the recommended way to handle multiline texts.

```python
data = TokenDataset("politics.csv")
```

You can also pass a list of texts to `texts` instead if you've processed them elsewhere.

```python
data = TokenDataset(texts = ["Lorem", "Ipsum", "Dolor"])
```

## Saving/Loading a TokenDataset

When creating a TokenDataset, you can automatically save it as a gzipped MessagePack binary when completed.

```python
data = TokenDataset("shakespeare.txt", save_cache=True)
```

Or save it after you've loaded it with the `save()` function.

```python
data = TokenDataset("shakespeare.txt")
data.save()
```

By default, it will save to `dataset_cache.tar.gz`. You can then reload that into another Python session by specifying the cache.

```python
data = TokenDataset("dataset_cache.tar.gz", from_cache=True)
```

## Using TokenDatasets withh a Custom GPT-2 Model

The default TokenDataset has a `block_size` of `1024`, which corresponds to the _maximum length of a GPT-2 output sequence_. If you're using a custom model w/ a different maximum. Additionally, you must explicitly provide the tokenizer, as the tokenizer will be different than the normal GPT-2 one.

## Merging TokenDatasets

Merging processed TokenDatasets can be done with the `merge_datasets()` function. By default, it will take samples equal to the smallest dataset from each TokenDataset, randomly sampling the appropriate number of texts from the larger datasets. This will ensure that model training does not bias toward one particular dataset. (it can be disabled by setting `equalize=False`)

(You _can_ merge bulk datasets and line-by-line datasets, but the training output may be bizarre!)

```python
from aitextgen.TokenDataset import TokenDataset, merge_datasets

data1 = TokenDataset("politics.csv")   # 1000 samples
data2 = TokenDataset("politics.csv")   # 4000 samples

data_merged = merge_datasets([data1, data2])   # 2000 samples
```
