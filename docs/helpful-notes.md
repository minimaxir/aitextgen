# Helpful Notes

A few helpful tips and tricks for using aitextgen.

<!--prettier-ignore-->
- **Not all AI generated text will be good**, hence why human curation is currently a necessary strategy for many finetuned models. In testing, only **5% — 10% of generated text** is viable. One of the design goals of aitextgen is to help provide tools to improve that signal-to-noise ratio.
- **You may not necessarily get better results with larger models**. Larger models perform better on academic benchmarks, yes, but the _quality_ of text can vary strongly depending on the size of the model used, _especially_ if you do not have a lot of input data.
- To convert a GPT-2 model trained using earlier TensorFlow-based finetuning tools such as gpt-2-simple to the PyTorch format, use the transformers-cli command and the [instructions here](https://huggingface.co/transformers/converting_tensorflow_models.html) to convert the checkpoint (where `OPENAI_GPT2_CHECKPOINT_PATH` is the _folder_ containing the model)
- When running on Google Cloud Platform (including Google Colab), it's recommended to download the TF-based GPT-2 from the Google API vs. downloading the PyTorch GPT-2 from Huggingface as the download will be _much_ faster and also saves Huggingface some bandwidth.
- If you want to generate text from a GPU, you must manually move the model to the GPU (it will not be done automatically to save GPU VRAM for training). Either call `to_gpu=True` when loading the model or call `to_gpu()` from the aitextgen object.
- Encoding your text dataset before moving it to a cloud/remote server is _strongly_ recommended. You can do that quickly from the CLI (`aitextgen encode text.txt`) Thanks to a few tricks, the file size is reduced by about 1/2 to 2/3, and the encoded text will instantly load on the remote server!
- If you're making a micro-GPT-2 model, using a GPU with a large batch size is recommended, as with the AdamW optimizer, it effectively has built-in batch normalization.
- If you make frequent use of the Colab Notebooks, I recommend purchasing [Colab Pro](https://colab.research.google.com/signup): the timeouts are infrequent, and being able to reliably train on a P100 (normally [$1.46/hr](https://cloud.google.com/compute/gpus-pricing)) for hours on end very quickly pays for itself!
