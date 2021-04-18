# Generating From GPT-2 1.5B

<!-- prettier-ignore -->
Want to generate a ton of text with the largest GPT-2 model, with the generation control provided by aitextgen? Now you can, at a surprisingly low cost! ($0.382/hr, prorated to the nearest second)

Here's how to set it up on Google Cloud Platform.

## Setting Up an AI Platform Notebook

An [AI Platform Notebook](https://cloud.google.com/ai-platform-notebooks) is a hosted Jupyter Lab instance on Google Cloud Platform oriented for AI training and inference. Since it requires zero setup _and has no additional costs outside of CPU/GPUs_, it's the best tool to play with aitextgen.

First, go to [AI Platform Notebooks in the GCP console](https://console.cloud.google.com/ai-platform/notebooks/) (if you haven't made a project + billing, it should prompt you to do so). Go to `New Instance`, select `PyTorch 1.4` and `With 1 NVIDIA Tesla T4`.

<!-- prettier-ignore -->
!!! note "Quotas"
    You may need T4 quota to create a VM with a T4; accounts should have enough by default, but you may want to confirm.

The rest of the VM config settings are fine to leave as/is, however make sure you check `Install NVIDIA GPU driver automatically for me`!

Once the instance is created, wait a bit (it takes awhile to install the driver), and a `OPEN JUPYTERLAB` button will appear. Click it to open the hosted Jupyter Lab

## Installing aitextgen on the VM

Now we have to install the dependencies, which only have to be done once.

In the Jupyter Lab instance, open a Terminal tab, and install both aitextgen and tensorflow (we'll need tensorflow later)

```sh
pip3 install aitextgen tensorflow
```

Now the harder part: we need to install and compile [apex](https://github.com/NVIDIA/apex) for FP16 support with the T4 GPU. To do that, run:

```sh
git clone https://github.com/NVIDIA/apex
cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

That will take a few minutes, but once that is done, you are good to go and do not need to rerun these steps again!

## Loading GPT-2 1.5B

Now go back to the Launcher and create a Python 3 Notebook (or upload the one here).

<!-- prettier-ignore -->
!!! warning "CUDA"
    You may want to ensure the Notebook sees the CUDA installation, which appears to be somewhat random. This can be verified by running `import torch` in a cell, then `torch.cuda.is_available()`.

In a cell, load aitextgen:

```py3
from aitextgen import aitextgen
```

In another cell, input and run:

```py3
ai = aitextgen(tf_gpt2="1558M", to_gpu=True, to_fp16=True)
```

A few things going on here:

- The TensorFlow-based GPT-2 1.5B is downloaded from Google's servers. (download rate is _very_ fast). This download will only occur once.
- It is converted to a corresponding PyTorch model, and then loaded.
- After it is loaded, it is converted to a FP16 representation.
- Then it is moved to the T4 GPU.

## Generating from GPT-2 1.5B

Now we can generate texts! The T4, for GPT-2 1.5B in FP16 mode, can generate about 30 texts in a batch without going OOM. (you can verify GPU memory usage at any time by opening up a Terminal and running `nvidia-smi`)

Create a cell and add:

```py3
ai.generate_to_file(n=300, batch_size=30)
```

<!-- prettier-ignore -->
!!! warning "Batch Size"
    The batch size of 30 above assumes the default `max_length` of 256. If you want to use the full 1024 token max length, lower the batch size to 15, as the GPU will go OOM otherwise.

And it will generate the texts to a file! When completed, you can double-click to view it in Jupyter Lab, and you can download the file by right-clicking it from the file viewer.

More importantly, all parameters to `generate` are valid, allowing massive flexibility!

```py3
ai.generate_to_file(n=150, batch_size=15, max_length=1024, top_p=0.9, temperature=1.2, prompt="President Donald Trump has magically transformed into a unicorn.")
```

## Cleanup

**Make sure you Stop the instance when you are done to avoid being charged**. To do that, go back to the AI Platform Notebook console, select the instance, and press Stop.

Since 100GB of storage may be pricy, you may want to consider deleting the VM fully if you are done with it as well.
