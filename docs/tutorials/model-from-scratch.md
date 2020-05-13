# Training a GPT-2 Model From Scratch

The original GPT-2 model released by OpenAI was trained on English webpages linked to from Reddit, with a strong bias toward longform content (multiple paragraphs).

If that is _not_ your use case, you may get a better generation quality _and_ speed by training your own model and Tokenizer. Examples of good use cases:

- Short-form content (e.g. Tweets, Reddit post titles)
- Non-English Text
- Encoded Text

It still will require a _massive_ amount of training time (several hours, even on a TPU), but will be more flexible.

## Building a Custom Tokenizer.

The `train
