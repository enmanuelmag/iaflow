---
sidebar_position: 3
---

# Management datasets

The instance created by the [Main class](./main-class) allow to manage the datasets, in this section we will see the CRUD for datasets.

## Create dataset

The `add_dataset` method is used to define a dataset. Has the following parameters:

- **name** (string): Name of the dataset.
- **epochs** (int): Number of epochs to train the model.
- **train_ds** (tf.data.Dataset | List): Dataset to train the model.
- **val_ds** (tf.data.Dataset | List, optional): Dataset to validate the model.
- **test_ds** (tf.data.Dataset | List, optional): Dataset to test the model.
- **batch_size** (int, optional): Batch size to train the model.
- **shuffle_buffer** (int, optional): Buffer size to shuffle the dataset.

```python title=Example
all_data = tf.data.Dataset.from_tensor_slices((
  tf.random.uniform([1000, 2]),
  tf.random.uniform([1000, 1])
))

train_ds = all_data.take(5)
val_ds = all_data.skip(5)

ia_maker.add_dataset(
  name='dataset_1',
  epochs=10,
  batch_size=32,
  shuffle_buffer=512,
  train_ds=train_ds,
  val_ds=val_ds
)
```

```bash title=Output
Dataset dataset_1 was added
```

## Update dataset

The `update_dataset` method is used to update a dataset. Has the following parameters:

- **name** (string): Name of the dataset to update.
- **epochs** (int): New number of epochs to train the model.
- **train_ds** (tf.data.Dataset | List): New dataset to train the model.
- **val_ds** (tf.data.Dataset | List): New dataset to validate the model.
- **test_ds** (tf.data.Dataset | List): New dataset to test the model.
- **batch_size** (int): New batch size to train the model.
- **shuffle_buffer** (int): New buffer size to shuffle the dataset.

```python title=Example
ia_maker.update_dataset(
  name='dataset_1',
  epochs=50,
  batch_size=128,
  shuffle_buffer=1024,
  train_ds=train_ds_2,
  val_ds=val_ds_2
)
```

```bash title=Output
Dataset dataset_1 was updated
```

## Delete dataset

The `delete_dataset` method is used to delete a dataset. Has the following parameters:

- **name** (string): Name of the dataset to delete.

```python title=Example
ia_maker.delete_dataset(name='dataset_1')
```

```bash title=Output
Dataset dataset_1 was deleted
```
