# Model Saving

There are are multiple ways to save models.

Whenever a model is saved, two files are generated: `pytorch_model.bin` which contains the model weights, and `config.json` which is needed to load the model if it is not the base 124M GPT-2.

Assuming we have an aitextgen model `ai`:

## Ad Hoc saving

The aitextgen model can be saved at any time using `save`.

```python
ai.save()
```

## Save to Google Drive

If you are using Google Colaboratory, you can mount your personal Google Drive to the notebook and save your models there.

<!-- prettier-ignore -->
!!! note "Downloading models from Colaboratory"
    It's strongly recommended to move models to Google Drive before downloading them from Colaboratory.

<!-- prettier-ignore-end -->

First mount your Google Drive using `mount_gdrive()`:

```python
from aitextgen.colab import mount_gdrive, copy_file_to_gdrive
mount_gdrive()
```

You'll be asked for an auth code; input it and press enter, and a `My Drive` folder will appear in Colab Files view.

You can drag and drop the model files into the Google Drive, or use `copy_file_to_gdrive` to copy them programmatically.

```python
copy_file_to_gdrive("pytorch_model.bin")
copy_file_to_gdrive("config.json")
```

## Saving During Training

By default, the `train()` function has `save_every = 1000`, which means the model will save every 1000 steps to the specified `output_dir` (`trained_model` by default). You can adjust as necessary.

## Saving During Training in Google Colab

Concerned about timeouts in Google Colab? aitextgen has a feature that will copy models to your Google Drive periodically in case the instance gets killed!

As long as your drive is mounted as above, pass `save_gdrive = True` to the `train() function:

```python
ai.train(save_gdrive=True)
```

This will save the model to the folder corresponding to the training `run_id` parameter (the datetime training was called, to prevent accidently overwriting).
