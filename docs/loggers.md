# Loggers

You can create loggers with popular tools such as TensorBoard and Weights and Biases by leveraging pytorch-lightning's logger functionality.

[See their documentation](https://pytorch-lightning.readthedocs.io/en/stable/loggers.html) on all the available options for loggers.

For example, if you want to create a TensorBoard logger, you can create it:

```python
from pytorch_lightning import loggers

tb_logger = loggers.TensorBoardLogger('logs/')
```

Then pass it to the `loggers` parameter for `ai.train()`.

```python
ai.train(train_data=data, loggers=tb_logger)
```
