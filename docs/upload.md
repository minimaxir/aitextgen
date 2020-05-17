# Upload Model to Huggingface

You can [upload your trained models](https://huggingface.co/transformers/model_sharing.html) to Huggingface, where it can be downloaded by others!

To upload your model, you'll have to create a folder which has **6** files:

- pytorch_model.bin
- config.json
- vocab.json
- merges.txt
- special_tokens_map.json
- tokenizer_config.json

You can generate all of these files at the same time into a given folder by running `ai.save_for_upload(model_name)`.

Then, follow the `transformers-cli` instructions to upload the model.

```sh
transformers-cli login
```

```sh
transformers-cli upload model_name
```

You (or another user) can download cache, and generate from that model via:

```
ai = aitextgen(model="username/model_name")
```
