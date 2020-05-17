# Command-Line Interface

aitextgen has a command-line interface to quickly automate common tasks, and make it less necessary to use a script; helpful if running on a remote server.

## Encode

Encodes given text `text.txt` into a cache and compressed `TokenDataset`, good for prepping a dataset for transit to a remote server.

```sh
aitextgen encode text.txt
```

If you are encoding a CSV, you should pass in the `line_by_line` parameter as well.

```sh
aitextgen encode reddit.csv --line_by_line True
```

## Train

To train/finetune on the default 124M GPT-2, given text `text.txt` and all default parameters:

```sh
aitextgen train text.txt
```

If you are using a cached/compressed dataset that ends with `tar.gz` (e.g one created by the Encoding CLI command above), you can pass that to this function as well.

```sh
aitextgen train dataset_cache.tar.gz
```

Other parameters to the TokenDataset constructor can be used.

## Generate

Loads a model and generates to a file.

By default, it will generate 20 texts to the file, 1 at a time at temperature of 0.7.

```sh
aitextgen generate
```

You can print to console instead by passing `--to_file False`

```sh
aitextgen generate --prompt "I believe in unicorns because" --to_file False
```
