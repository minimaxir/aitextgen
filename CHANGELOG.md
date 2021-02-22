# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.4.0] - 2021-02-21

- Increased minimum versions of dependencies (`transformers` to 4.3.0, `pytorch-lightning` to 1.2.0)
  - Remove dependency on `tokenizers` as `transformers` pins it.
- Made Fast tokenizers the default (as it is the default in `transformers` 4.0.0)
- Made serialized tokenizers the default for custom tokenizers, and added support for loading them for both `aitextgen` and `TokenDataset`s
- Added gradient checkpointing for GPT-2, and set it to the default for training 355M and 774M.
- Added layer freezing to freeze the first `n` layers of GPT-2 while training. This allows 1.5B GPT-2 to be trained with a high `n`.
- Added schema-based generation for specificed schema_tokens (which can be encoded in the Transformers config). This can be used with an appropriate dataset for schema-based generation.
- Switched TensorFlow weight download URL from GCP (as OpenAI removed it from there) to Azure
- Fixed issue where prompt character length was used to check for a too-long assert instead of prompt token length (#90)
- Workaround breaking issue in Transformers 4.3.0 by moving special token stripping into aitextgen instead of the tokenizer (#90)
- Added an `lstrip` param to generation, which strips all whitespace at the beginning of generated text (related to point above)

## [0.3.0] - 2020-11-30

- Increased minimum versions of dependencies (`transformers` to 4.0.0, `pytorch-lightning` to 1.0.8, Pytorch to 1.6)
- Fixed imports to account for new Transfomers file architecture
- Fixed training to account for new transformer/pytorch-lightning minimums
- Fully removed TorchScript code (ONNX implementation will supercede it)
- Made prompt specification for generation more canonical with Transformers
- Set default `vocab` size for new tokenizers to `1000`
- Began work on serializing tokenizers in accordance to the new `tokenizers` approach

## [0.2.1] - 2020-06-28

### Added

- CHANGELOG.md

## [0.2.0] - 2020-06-01

### Added

- Progress bar for loading a dataset.
- `progress_bar_refresh_rate` parameter for `train()` and `TokenDataset()`.

### Changed

- Set numpy data store for `TokenDataset`.
- Set single-text files to be loaded delimited as newlines.

### Removed

- `shuffle` and `seed` parameters for TokenDataset.

## [0.1.1] - 2020-05-17

### Changed

- Set `generate()` defaults to `max_length=256` and `temperature=0.7`.
- Added to docs notes about GPT-2 maximum length of 1024.

## [0.1] - 2020-05-17

### Added

- Everything!
