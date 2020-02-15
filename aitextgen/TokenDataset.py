import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer
import logging
import csv
import os

logger = logging.getLogger(__name__)


class TokenDataset(Dataset):
    """
    Class that merges TextDataset and LineByLineTextDataset from
    run_language_modeling.py in transformers, plus
    adds more ways to ingest text such as with CSVs.

    For user simplicity, features are not cached.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer,
                 texts, line_by_line: bool, file_path, header,
                 block_size=512):

        # if texts are present, just tokenize them.
        if texts is not None:
            self.examples = tokenizer.batch_encode_plus(
                texts, add_special_tokens=True, max_length=block_size)["input_ids"]

        # if a file is specified, and it's line-delimited,
        # the text must be processed line-by-line
        elif line_by_line:
            assert os.path.isfile(file_path)

            texts = read_lines_from_file(file_path, header=header)

            self.examples = tokenizer.batch_encode_plus(
                texts, add_special_tokens=True, max_length=block_size)["input_ids"]

        # if a file is specified, and it's not line-delimited,
        # the texts must be parsed in chunks.
        else:
            assert os.path.isfile(file_path)

            block_size = block_size - (tokenizer.max_len -
                                       tokenizer.max_len_single_sentence)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                self.examples.append(tokenizer.build_inputs_with_special_tokens(
                    tokenized_text[i:i + block_size]))

        logger.info("{:,} samples loaded.".format(len(self.examples)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


def read_lines_from_file(file_path, header=True):
    """
    Retrieves texts from a newline-delimited file/CSV and returns as a list.
    """

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        if header:
            f.readline()
        if file_path.endswith('.csv'):
            reader = csv.reader(f)
            texts = [line for line in reader]
        else:
            texts = [line for line in f.read().splitlines()
                     if (len(line) > 0 and not line.isspace())]

    return texts
