# Upcoming Features

Here is a list of features, in no particular order, of what I *hope* to add to aitextgen, given that it's feasible.

* Must be TPU compatible (work with SavedModel + GCP gpu)
    * Must be able to train using Colab's free TPU, if possible
* Sparse Transformers?
* Have a super minimal version that can be distributed with a package. (< 3MB)
* Future-proofed
* Context support, provided as a named dict to a generation function
* Include keyword conditioning as well (w/ ray as a possible optional dependency)
* Can all run in a single session, do not need to reload session
* Allow a user-curated mode
* Allow probabilities of next token
* Add seed for batch generation?
* Cross-training / training schedules
* Sample weighting, so network can weight earlier text in the sequence stronger than later text (e.g. recent tweets than older tweets!)
* Replace tokens in a tokenizer, e.g. replace rare tokens in a tweet generator with <|startoftext|> to save space, or allow a manual replacement (store replacement dict in a JSON config
* Support any CLM model, not just GPT2.
* Support training model from scratch (will need to add tokenizer export)
* Include a small model trainable on a CPU, trained from a distilled larger model
* Allow excluding of tokens by index (e.g. allow model generation without the letter e)
* For postfix prediction, allow returning best guess or probabilities of all classes.
* For vocab prefix, use heuristics on tokens, since very speedy and can scale high with strong tokenizers
* Dimensionality by using activation weights of all tokens prior to EOS token.
    * Use dimensionality to calculate a similarity score from generated texts to real texts; scores below a threshold may be considered incoherent and can be discarded.
* Dedupe existing texts when generating using token id matching.
    * Allow users to load data outside of finetuning, which will be necessary for blending training.
* Async generation, for batch generation
* Tensorboard support as a parameter
* Allow cycling context tokens when generating
* Use ray for async generation. May need a custom generation function.
    * Use web sockets in starlette so output can be returned token by token.
* Use pytorch-lightning for easy TPU support
    * https://github.com/huggingface/transformers/pull/3053
* Unlimited text generation regardless of model window size.

## Completed Features

* Use ANSI escape sequences to color text output in terminal. Bold for prefix. Only output if printing to console, do not do if saving as a list.
* Pass tqdm progress bar instance to multigenerate