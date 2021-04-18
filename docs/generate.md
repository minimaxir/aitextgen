# Text Generation

Thanks to the base Transformers package, aitextgen has more options for generating text than other text-generating apps before.

## Generation Parameters

See [this article](https://huggingface.co/blog/how-to-generate) by Huggingface engineer Patrick von Platen for how sampling and these parameters are used in practice.

- `n`: Number of texts generated.
- `max_length`: Maximum length of the generated text (default: 200; for GPT-2, the maximum is 1024; for GPT Neo, the maximum is 2048)
- `prompt`: Prompt that starts the generated text and is included in the generated text.
- `temperature`: Controls the "craziness" of the text (default: 0.7)
- `top_k`: If nonzero, limits the sampled tokens to the top _k_ values. (default: 0)
- `top_p`: If nonzero, limits the sampled tokens to the cumulative probability

Some lesser-known-but-still-useful-parameters that are unique to Transformers:

<!--prettier-ignore-->
!!! warning "Performance"
    Enabling these parameters may slow down generation.

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

- `ai.generate()`: Generates and prints text to console. If `prompt` is used, the `prompt` is **bolded**.
- `ai.generate_one()`: A helper function which generates a single text and returns as a string (good for APIs)
- `ai.generate_samples()`: Generates multiple samples at specified temperatures: great for debugging.
- `ai.generate_to_file()`: Generates a bulk amount of texts to file. (this accepts a `batch_size` parameter which is useful if using on a GPU, as it can generate texts in parallel with no performance loss)

<!-- prettier-ignore -->
!!! note "lstrip and nonempty_output"
    By default, the `lstrip` and `nonempty_output` parameters to `generate` are set to `True`, which alters the behavior of the generated text in a way that is most likely preferable.  `lstrip`: Removes all whitespace at the beginning of the generated space. `nonempty_output`: If the output is empty (possible on shortform content), skip it if generating multiple texts, or try again if it's a single text. If `min_length` is specified, the same behavior occurs for texts below the minimum length after processing.

## Seed

aitextgen has a new `seed` parameter for generation. Using any generate function with a `seed` parameter (must be an integer) and all other models/parameters the same, and the generated text will be identical. This allows for reproducible generations in case someone accuses you of faking the AI output.

For `generate_to_file()`, the 8-digit number at the end of the file name will be the seed used to generate the file, making reprodicibility easy.
