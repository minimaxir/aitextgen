# Model Loading

There are several ways to load models.

<!--prettier-ignore-->
!!! note "Continue Training/Finetuning"
    You can further train a model by reloading a model that has already been trained, using the methods outlined below.

## Loading an aitextgen model

For the base case, loading the default 124M GPT-2 model via Huggingface:

```py3
ai = aitextgen()
```

The downloaded model will be downloaded to `cache_dir`: `/aitextgen` by default.

If you're loading a custom model for a different GPT-2/GPT-Neo architecture _from scratch_ but with the normal GPT-2 tokenizer, you can pass only a config.

```py3
from aitextgen.utils import GPT2ConfigCPU
config = GPT2ConfigCPU()
ai = aitextgen(config=config)
```

While training/finetuning a model, two files will be created: the `pytorch_model.bin` which contains the weights for the model, and a `config.json` illustrating the architecture for the model. Both of these files are needed to reload the model.

If you've finetuned a model using aitextgen (the default model), you can pass the **folder name** containing the generated `pytorch_model.bin` and `config.json` to aitextgen (e.g. `trained_model`, which is where trained models will be saved by default).

<!--prettier-ignore-->
!!! note "Same Directory"
    If both files are in the current directory, you can pass `model_folder="."`.

```py3
ai = aitextgen(model_folder="trained_model")
```

These examples assume you are using the default GPT-2 tokenizer. If you have a _custom tokenizer_, you'll need to pass that along with loading the model.

```py3
ai3 = aitextgen(model_folder="trained_model",
                tokenizer_file="aitextgen.tokenizer.json")
```

If you want to download an alternative GPT-2 model from Hugging Face's repository of models, pass that model name to `model`.

```py3
ai = aitextgen(model="minimaxir/hacker-news")
```

The model and associated config + tokenizer will be downloaded into `cache_dir`.

This can also be used to download the [pretrained GPT Neo models](https://huggingface.co/EleutherAI) from EleutherAI.

```py3
ai = aitextgen(model="EleutherAI/gpt-neo-125M")
```

## Loading TensorFlow-based GPT-2 models

aitextgen lets you download the models from Microsoft's servers that OpenAI had uploaded back when GPT-2 was first released in 2019. These models are then converted to a PyTorch format.

To use this workflow, pass the corresponding model number to `tf_gpt2`:

```py3
ai = aitextgen(tf_gpt2="124M")
```

This will cache the converted model locally in `cache_dir`, and using the same parameters will load the converted model.

The valid TF model names are `["124M","355M","774M","1558M"]`.
