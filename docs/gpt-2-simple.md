# Importing from gpt-2-simple

Want to import a model trained using [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple), or another GPT-2 based finetuning approach? You can do that [using the transformers-cli](https://huggingface.co/transformers/converting_tensorflow_models.html).

In the case of gpt-2-simple (where the output is structured `checkpoint/run1`), you'd `cd` into the directory containing the `checkpoint` folder and run:

```sh
transformers-cli convert --model_type gpt2 --tf_checkpoint checkpoint/run1 --pytorch_dump_output pytorch --config checkpoint/run1/hparams.json
```

This will put a `pytorch_model.bin` and `config.json` in the `pytorch` folder, which is what you'll need to pass to `aitextgen()` to load the model.

That's it!
