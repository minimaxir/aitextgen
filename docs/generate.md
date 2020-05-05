# Text Generation

Thanks to the base Transformers package, aitextgen has more options for generating text than other text-generating apps before.

## Generation Parameters

See this article by [Huggingface engineer](https://huggingface.co/blog/how-to-generate) Patrick von Platen for how sampling and these parameters are used in practice.

- `n`: Number of texts generated.
- `max_length`: Maximum length of the generated text (default: 200; for GPT-2, the maximum is 1024.)
- `prompt`: Prompt that starts the generated text and is included in the generate text. (used to be `prefix` in previous tools)
- `temperature`: Controls the "craziness" of the text (default: 0.7)
- `top_k`: If nonzero, limits the sampled tokens to the top _k_ values. (default: 40)
- `top_p`: If nonzero, limits the sampled tokens to the cumulative probability

## Generation Functions

Given a `aitextgen` object with a loaded model + tokenizer named `ai`:

- `ai.generate()`: Generates and prints text to console. If `prompt` is used, the `prompt` is bolded. (a la [Talk to Transformer](https://talktotransformer.com))
- `ai.generate_one()`: A helper function which generates a single text and returns as a string (good for APIs)
- `ai.generate_samples()`: Generates multiple samples at specified temperatures: great for debugging.
- `ai.generate_to_file()`: Generates a bulk amount of texts to file. (this accepts a `batch_size` parameter which is useful if using on a GPU)

## Seed

aitextgen has a new `seed` parameter for generation. Using any generate function with a `seed` parameter (must be an integer) and all other models/parameters the same, and the generated text will be identical. This allows for reproducible generations.

For `generate_to_file()`, the 8-digit number at the end of the file name will be the seed used to generate the file, making reprodicibility easy.
