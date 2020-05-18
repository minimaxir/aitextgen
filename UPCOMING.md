# Upcoming Features

Here is a list of features, in no particular order, of what I _hope_ to add to aitextgen, given that it's feasible.

## Training Features

- Sparse Transformers?
- Include keyword conditioning as well
- - For vocab prefix, use heuristics on tokens, since very speedy and can scale high with strong tokenizers
- Context support, provided as a named dict to a generation function
  e sequence stronger than later text (e.g. recent tweets than older tweets!)

## Generation Features

- Allow a user-curated mode
- Allow returning probabilities of next token
- - Console BG coloring of token prediction confidence?
- Allow excluding of tokens by index (e.g. allow model generation without the letter e)
- For postfix prediction, allow returning best guess or probabilities of all classes.
- Calculate text Dimensionality by using activation weights of all tokens prior to EOS token.
  - Use dimensionality to calculate a similarity score from generated texts to real texts; scores below a threshold may be considered incoherent and can be discarded.
- Dedupe existing texts when generating using token id matching.
- Allow cycling context tokens when generating
- Unlimited text generation via sliding context window (need to include a warning if doing so)

## Deployment Features

- Use [ray](https://github.com/ray-project/ray) async actors for async generation. May need a custom async generation function.
  - Use websockets in [starlette](https://www.starlette.io) so output can be returned token by token.
- Export function: PyTorch trace, TensorFlow Serving, TensorFlow.js, CoreML

## Quality-of-Life Features

- Have a super minimal version that can be distributed with the PyPi package. (< 3MB)
  - Include a small model trainable on a CPU, trained from a distilled larger model
- Support any CLM model, not just GPT2.
- Support CPU-based XLA for faster PyTorch CPU Train/Predict performance
