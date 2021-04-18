# Colaboratory Notebooks

You cannot finetune OpenAI's GPT-2 models on CPU (and not even on some consumer GPUs). Therefore, there are a couple Google Colaboratory notebooks, which provide a GPU suitable for finetuning a model.

The Colab Notebooks also contain utilities to make it easier to export the model to Google Drive during and after training.

## Finetuning OpenAI's Model

[Colab Notebook](https://colab.research.google.com/drive/15qBZx5y9rdaQSyWpsreMDnTiZ5IlN0zD?usp=sharing)

A Notebook for finetuning OpenAI's model on a GPU. This is the most common use case.

<!-- prettier-ignore -->
!!! note
    Currently you can only finetune the 124M/355M/774M OpenAI GPT-2 models, with the latter two forcing `gradient_checkpointing=True` to ensure it does not cause the Colab GPU to go OOM.

## Training Your Own GPT-2 Model

[Colab Notebook](https://colab.research.google.com/drive/144MdX5aLqrQ3-YW-po81CQMrD6kpgpYh?usp=sharing)

A Notebook for creating your own GPT-2 model with your own tokenizer. See the Model From Scratch section on the advantages and disadvantages of this approach.
