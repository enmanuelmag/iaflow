---
sidebar_position: 4
---

# Train model

The instance created by the [Main class](./main-class) allow to train a model, in this section we will see how to use the `train` method.

The `train` method is used to train a model. Has the following parameters:

- **model_data** (dictionary): Data of the model to train, this data is returned by the `add_model` method.
- **dataset_name** (string): Name of the dataset to train the model.
- **epochs** (int, optional): Number of epochs to train the model. By default will be used the value defined in the `epochs` parameter of the `add_dataset` method.
- **batch_size** (int, optional): Batch size to train the model. By default will be used the value defined in the `batch_size` parameter of the `add_dataset` method.
- **initial_epoch** (int, optional): Initial epoch to train the model. By default is 0.
- **shuffle_buffer** (int, optional): Size of the shuffle buffer. By default will be used the value defined in the `shuffle_buffer` parameter of the `add_dataset` method.
- **force_creation** (bool, optional): If is `True` the model will be create again even if the model already exists. By default is `False`.
- **train_ds** (tf.data.Dataset | List, optional): Dataset to train the model. By default will be used the dataset defined in the `add_dataset` method.
- **val_ds** (tf.data.Dataset | List, optional): Dataset to validate the model. By default will be used the dataset defined in the `add_dataset` method.

```python title=Example
ia_maker.train(
  model_1_data,
  epochs=5,
  dataset_name='dataset_1'
)
```

:::info
If a model is already trained and you call the `train` method again, the model will be loaded and continue to train, but don't forget to send the `initial_epoch` parameter with the number of epochs already trained.
:::