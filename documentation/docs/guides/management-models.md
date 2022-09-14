---
sidebar_position: 1
---

# Management models

## Usage

This library has a maser class called `IAFlow`, that has functions to management model creation and training. In this section, we will see the CRUD for models.

### Constructor

The constructor of `IAFlow` class has the following parameters:

- **models_folder**: Folder to save the models.
- **params_notifier**: Parameters for notifier, see documentation [here](https://pypi.org/project/notify-function/#description).
- **builder_function**: Function to build the model.

```python
from iaflow import IAFlow

def custom_builder(input_shape):
  model = Sequential([
    Dense(units=512, activation='relu'),
    Dense(units=512, activation='relu'),
    Dense(units=1, activation='sigmoid')
  ])
  return model

# Parameters for notifier, see documentation
# https://pypi.org/project/notify-function/#description
params_notifier = {
  'title': 'Training update',
  'webhook_url': os.environ.get('WEBHOOK_URL'),
  'frequency_epoch': 20 # Send a notification every 20 epochs, by default it is every epoch
}

ia_maker = IAFlow(
  models_folder='./models',
  params_notifier=params_notifier,
  builder_function=custom_builder
)
```

## Add a model

The `add_model` method is used to define a model. Has the following parameters:

- **model_name** (string): Name of the model.
- **run_id** (string): Run id of the model, by default is the current time in format `YYYY.MM.DD_HH.mm.ss`.
- **model_params** (dictionary): Parameters to build the model, this parameters will be passed to the `builder_function` defined in the constructor.
- **load_model_params** (dictionary): Parameters to load the model, see documentation of `tf.keras.models.load_model`.
- **compile_params** (dictionary): Parameters to compile the model, see documentation of `tf.keras.Model.compile`.

```python
model_1_data = ia_maker.add_model(
  model_name='model_1',
  model_params={ 'input_shape': (2, 1) },function
  load_model_params={},documentation of tf.keras.models.load_model
  compile_params={
    'optimizer': 'adam', 'loss': 'mse',
    'metrics': ['accuracy']
  },
)
```

Returns:
```
{
  'run_id': A unique identifier for the model,
  'model_name': Name of the model,
  'model_ident': Name of the model and params,
  'model_params': Parameters to build the model,
  'compile_params': Parameters to compile the model,
  'load_model_params': Parameters to load the model,
  'check_path': Path to save the checkpoints,
  'path_model': Path to save the model,
  'log_dir': Path to save the logs,
}
```


Output:
```
GPU info: /physical_device:GPU:0

Model model_1/2022.07.18_10.04.03 added
```

## Show models

The `show_models` method is used to show the models defined via stantard output.

```python
print('Show all models:')
ia_maker.show_models()
```
Output:
```bash
Show all models:
Models:
  model_1
    2022.07.18_10.04.03
      model_ident: model_1_(2, 1)
      model_params: {'input_shape': (2, 1)}
      compile_params: {'optimizer': 'adam', 'loss': 'mse', 'metrics': ['accuracy']}
      load_model_params: {'filepath': './models/model_1/2022.07.18_10.04.03/model_1_(2, 1)_checkpoint.h5'}
      check_path: ./models/model_1/2022.07.18_10.04.03/model_1_(2, 1)_checkpoint.h5
      path_model: ./models/model_1/2022.07.18_10.04.03
      log_dir: ./models/model_1/2022.07.18_10.04.03/logs
```

## Update a model
The `update_model` method is used to update a model. Has the following parameters:

- **model_name** (string): Name of the model.
- **run_id** (string): Run id of the model.
- **model_params** (dictionary): to update the parameters to build the model.
- **compile_params** (dictionary): to update the parameters to compile the model.
- **load_model_params** (dictionary): to update the parameters to load the model.

```python
ia_maker.update_model(
  model_name='model_1',
  run_id='2022.07.18_10.04.03',
  model_params={ 'input_shape': (2, 1) },
  compile_params={
    'optimizer': 'adam', 'loss': 'mse',
    'metrics': ['accuracy']
  },
  load_model_params={}
)
```

Output:
```
Model model_1/modelo_dense_1 was updated
```

## Delete a model

The `delete_model` method is used to delete a model. Has the following parameters:

- **model_data** (string): Model data returned by `add_model` method.
- **delete_folder** (boolean): Delete the folder of the model, by default is `False`.

```python
ia_maker.delete_model(model_1_data, delete_folder=True)
```

Output:
```
Model model_1/2022.07.18_10.04.03 was deleted
```
