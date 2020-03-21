import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer
import logging
import csv
import os
import msgpack
import random

logger = logging.getLogger(__name__)


class TokenDataset(Dataset):
    """
    Class that merges TextDataset and LineByLineTextDataset from
    run_language_modeling.py in transformers, plus
    adds more ways to ingest text such as with CSVs.

    ## Parameters

    * **tokenizer**: Tokenizer for the corresponding model.
    * **texts**: A list of input texts (if providing texts manually)
    * **file_path**: A string indicating the relative file path of the text
    to be tokenized.
    * **line_by_line**: A boolean to indicate if the input file should be read
    line by line (True) or as a full text (False).
    * **from_cache**: A string indicating if loading from a pregenerated MsgPack
    dump.
    * **header**: A boolean indicating if loading from a CSV, if it has a header.
    * **save_cache**: A boolean indicating whether to save the tokenized
    dataset as a MsgPack dump to load later.
    * **cache_destination**: A string indicating where to save the cache.
    * **block_size**: An integer indicating maximum length of the text document
    (usually set by the model architecture)
    * **tokenized_texts**: Texts that are already tokenized; only should
    be used by merge_datasets().
    """

    def __init__(
        self,
        tokenizer=None,
        texts=None,
        file_path=None,
        line_by_line=None,
        from_cache=False,
        header=True,
        save_cache=False,
        cache_destination=None,
        block_size=1024,
        tokenized_texts=None,
    ):

        # Special case; load tokenized texts immediately
        if tokenized_texts:
            self.examples = tokenized_texts
            self.str_suffix = "by merging TokenDatasets."
            return

        assert any([texts, file_path]), "texts or file_path must be specified."
        if not from_cache:
            assert tokenizer is not None, "A tokenizer must be specified."

        # If a cache path is provided, load it.
        if from_cache:
            with open(file_path, "rb") as f:
                self.examples = msgpack.unpack(f)
            self.str_suffix = "via cache."

        # if texts are present, just tokenize them.
        elif texts is not None:
            self.examples = tokenizer.batch_encode_plus(
                texts, add_special_tokens=True, max_length=block_size
            )["input_ids"]
            self.str_suffix = "via application."

        # if a file is specified, and it's line-delimited,
        # the text must be processed line-by-line
        elif line_by_line:
            assert os.path.isfile(file_path)

            texts = read_lines_from_file(file_path, header=header)

            self.examples = tokenizer.batch_encode_plus(
                texts, add_special_tokens=True, max_length=block_size
            )["input_ids"]

            self.str_suffix = f"from line-by-line file at {file_path}."

        # if a file is specified, and it's not line-delimited,
        # the texts must be parsed in chunks.
        else:
            assert os.path.isfile(file_path)

            block_size = block_size - (
                tokenizer.max_len - tokenizer.max_len_single_sentence
            )

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                self.examples.append(
                    tokenizer.build_inputs_with_special_tokens(
                        tokenized_text[i : i + block_size]
                    )
                )

            self.str_suffix = f"from file at {file_path}."

        logger.info("{:,} samples loaded.".format(len(self.examples)))

        if save_cache:
            self.save(cache_destination)

    def save(self, cache_destination="model_cache.msgpack"):
        assert len(self.examples) > 0, "No data loaded to save."

        logger.info("Caching dataset to {}".format(cache_destination))

        with open(cache_destination, "wb") as f:
            msgpack.pack(self.examples, f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

    def __repr__(self):
        return f"TokenDataset containing {len(self.examples):,} examples loaded {self.str_suffix}"


def read_lines_from_file(file_path, header=True):
    """
    Retrieves texts from a newline-delimited file/CSV and returns as a list.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        if header:
            f.readline()
        if file_path.endswith(".csv"):
            reader = csv.reader(f)
            texts = [str(line) for line in reader]
        else:
            texts = [
                str(line)
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]

    return texts


def merge_datasets(datasets, equalize=True, seed=None):
    """
    Merges multiple TokenDatasets into a single TokenDataset.
    This assumes that you are using the same tokenizer for all TokenDatasets.

    ## Parameters

    * **datasets**: A list of TokenDatasets.
    * **equalize**: Whether to take an equal amount of samples from all
    input datasets (by taking random samples from each dataset equal to the smallest dataset) in order to balance out the result dataset.
    * **seed**: Seed to control the random sampling, if using equalize.
    """

    len_smallest = min([len(dataset) for dataset in datasets])

    if seed:
        random.seed(seed)

    tokenized_texts = []

    for dataset in datasets:
        if equalize:
            texts_subset = random.sample(dataset.examples, len_smallest)
            tokenized_texts.append(texts_subset)
        else:
            tokenized_texts.append(dataset.examples)

    # Reset seed
    if seed:
        random.seed()

    return TokenDataset(tokenized_texts=tokenized_texts)
