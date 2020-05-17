# Text Generation

Thanks to the base Transformers package, aitextgen has more options for generating text than other text-generating apps before.

## Generation Parameters

See [this article](https://huggingface.co/blog/how-to-generate) by Huggingface engineer Patrick von Platen for how sampling and these parameters are used in practice.

- `n`: Number of texts generated.
- `max_length`: Maximum length of the generated text (default: 200; for GPT-2, the maximum is 1024.)
- `prompt`: Prompt that starts the generated text and is included in the generate text. (used to be `prefix` in previous tools)
- `temperature`: Controls the "craziness" of the text (default: 0.7)
- `top_k`: If nonzero, limits the sampled tokens to the top _k_ values. (default: 0)
- `top_p`: If nonzero, limits the sampled tokens to the cumulative probability

Some lesser-known-but-still-useful-parameters that are unique to Transformers:

- `num_beams`: If greater than 1, executes beam search for cleaner text.
- `repetition_penalty`: If greater than 1.0, penalizes repetition in a text to avoid infinite loops.
- `length_penalty`: If greater than 1.0, penalizes text proportional to the length
- `no_repeat_ngram_size`: Token length to avoid repeating given phrases.

## Generation Functions

Given a `aitextgen` object with a loaded model + tokenizer named `ai`:

<!--prettier-ignore-->
!!! note "About devices"
    aitextgen does not automatically set the device used to generate text. If you
    want to generate on the GPU, make sure you call `ai.to_gpu()` beforehand, or
    load the model into the GPU using `ai = aitextgen(to_gpu=True)`

- `ai.generate()`: Generates and prints text to console. If `prompt` is used, the `prompt` is bolded. (a la [Talk to Transformer](https://talktotransformer.com))
- `ai.generate_one()`: A helper function which generates a single text and returns as a string (good for APIs)
- `ai.generate_samples()`: Generates multiple samples at specified temperatures: great for debugging.
- `ai.generate_to_file()`: Generates a bulk amount of texts to file. (this accepts a `batch_size` parameter which is useful if using on a GPU)

<!-- prettier-ignore -->
!!! note "Cleanup"
    By default, the `cleanup` parameter is set to True, which automatically removes texts that are blatantly malformed (e.g. only 2 characters long). Therefore, there may be less than `n` results returned. You can disabled this behavior by setting `cleanup=False`.

## Seed

aitextgen has a new `seed` parameter for generation. Using any generate function with a `seed` parameter (must be an integer) and all other models/parameters the same, and the generated text will be identical. This allows for reproducible generations.

For `generate_to_file()`, the 8-digit number at the end of the file name will be the seed used to generate the file, making reprodicibility easy.
