# Improving Generation Performance

A few tips and tricks for improving generation performance for both on CPUs and GPUs. (note that with these tricks, you cannot train the model afterwards!)

## CPU

### Quantization

PyTorch has the ability to quantize models on the CPU. Currently, it will only quantize the Linear layer of GPT-2, but the generation performances increases **15% — 25%**; far from trivial!

To quantize a model after it's loaded, just run:

```py3
ai.quantize()
```

## GPU

### FP16

Certain GPUs, notably the cheap T4 and the expensive V100, support the ability to process models using FP16, giving massive speed and memory improvements,

Assuming you are using a compatable GPU and already have [apex](https://github.com/NVIDIA/apex) installed, you can convert a model to the "half" FP16 mode with this:

```py3
ai.to_fp16()
```

If you want to convert the model _before_ loading it into GPU memory (which may help avoid memory leaks), you can instantiate the model like this:

```py3
ai.to_fp16(to_gpu=True, to_fp16=True)
```

With this, you can generate massive amounts of text from even the GPT-2 1.5B model!
