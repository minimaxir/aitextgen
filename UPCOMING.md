# Upcoming Features

Here is a list of features, in no particular order, of what I *hope* to add to aitextgen, given that it's feasible.

## Training Features

* Must be TPU compatible
    * Must be able to train using Colab's free TPU, if possible
* Sparse Transformers?
* Include keyword conditioning as well
* * For vocab prefix, use heuristics on tokens, since very speedy and can scale high with strong tokenizers
* Cross-training multiple distinct text datasets
* Context support, provided as a named dict to a generation function
* Sample weighting, so network can weight earlier text in the sequence stronger than later text (e.g. recent tweets than older tweets!)
* Replace tokens in a tokenizer, e.g. replace rare tokens in a tweet generator with <|startoftext|> to save space, or allow a manual replacement (store replacement dict in a JSON config)
* Use pytorch-lightning for easy TPU support (https://github.com/huggingface/transformers/pull/3053)
* Tensorboard support as a parameter

## Generation Features

* Allow a user-curated mode
* Allow probabilities of next token
* Allow excluding of tokens by index (e.g. allow model generation without the letter e)
* For postfix prediction, allow returning best guess or probabilities of all classes.
* Calculate text Dimensionality by using activation weights of all tokens prior to EOS token.
    * Use dimensionality to calculate a similarity score from generated texts to real texts; scores below a threshold may be considered incoherent and can be discarded.
* Dedupe existing texts when generating using token id matching.
* Allow cycling context tokens when generating

## Deployment Features

* Use [ray](https://github.com/ray-project/ray) async actors for async generation. May need a custom async generation function.
    * Use websockets in [starlette](https://www.starlette.io) so output can be returned token by token.
* Export function: PyTorch trace, TensorFlow Serving, TensorFlow.js, CoreML

## Quality-of-Life Features

* Can all run in a single session, do not need to reload session
* Have a super minimal version that can be distributed with the PyPi package. (< 3MB)
  * Include a small model trainable on a CPU, trained from a distilled larger model
* Support any CLM model, not just GPT2.
* Support training model from scratch (will need to add tokenizer export)
* Unlimited text generation regardless of model window size.
* Support CPU-based XLA for faster PyTorch CPU Train/Predict performance
