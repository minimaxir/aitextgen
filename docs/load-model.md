# Model Loading

There are several ways to load models.

## Loading an aitextgen model

The closer to the default 124M GPT-2 model, the fewer files you need!

For the base case, loading the default model via Huggingface:

```python
ai = aitextgen()
```

The downloaded model will be downloaded to `cache_dir`: `/aitextgen` by default.

If you've finetuned a 124M GPT-2 model using aitextgen, you can pass the generated `pytorch_model.bin` to aitextgen:

```python
ai = aitextgen(model="pytorch_model.bin")
```

If you're loading a finetuned model of a different GPT-2 architecture, you'll must also pass the generated `config.json` to aitextgen:

```python
ai = aitextgen(model="pytorch_model.bin", config=config)
```

If you want to download an alternative GPT-2 model from Huggingface's repository of models, pass that model name to `model`.

```python
ai = aitextgen(model="minimaxir/hacker-news")
```

The model and associated config + tokenizer will be downloaded into `cache_dir`.

### Loading TensorFlow-based GPT-2 models

aitextgen lets you download the models from Google's servers that OpenAI had uploaded back when GPT-2 was first released in 2019. These models are then converted to a PyTorch format.

It's counterintuitive, but it's _substantially_ faster than downloading from Huggingface's servers, especially if you are running your code on Google Cloud Platform (e.g. Colab notebooks)

To use this workflow, pass the corresponding model number to `tf_gpt2`:

```python
ai = aitextgen(tf_gpt2="124M")
```

This will cache the converted model locally in `cache_dir`, and using the same parameters will load the converted model.

The valid TF model names are `["124M","355M","774M","1558M"]`.
