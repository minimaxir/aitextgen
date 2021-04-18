# Loggers

You can create loggers with popular tools such as [TensorBoard](https://www.tensorflow.org/tensorboard) and [Weights and Biases](https://www.wandb.com) by leveraging pytorch-lightning's logger functionality.

[See their documentation](https://pytorch-lightning.readthedocs.io/en/stable/loggers.html) on all the available options for loggers.

For example, if you want to create a TensorBoard logger, you can create it:

```py3
from pytorch_lightning import loggers

tb_logger = loggers.TensorBoardLogger('logs/')
```

Then pass it to the `loggers` parameter for `ai.train()`.

```py3
ai.train(train_data=data, loggers=tb_logger)
```
