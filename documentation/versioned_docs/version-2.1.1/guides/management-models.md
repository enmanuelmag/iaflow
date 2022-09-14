---
sidebar_position: 2
---

# Management models

The instance created by the [Main class](./main-class) allow to manage the models, in this section we will see the CRUD for models.

## Create a model

The `add_model` method is used to define a model. Has the following parameters:

- **model_name** (string): Name of the model.
- **run_id** (string): Run id of the model, by default is the current time in format `YYYY.MM.DD_HH.mm.ss`.
- **model_params** (dictionary): Parameters to build the model, this parameters will be passed to the `builder_function` defined in the constructor.
- **load_model_params** (dictionary, optional): Parameters to load the model, see documentation of `tf.keras.models.load_model`.
- **compile_params** (dictionary, optional): Parameters to compile the model, see documentation of `tf.keras.Model.compile`.

The added model will be saved in the following structure folder:

```txt title="Folder structure"
models_folder
└─ model_name
   └─ run_id
      ├─ logs                               
      │  ├─ train
      │  ├─ validation
      ├─ model_name_model_params_checkpoint.h5
      └─ model_name_params.json
```


```python title="Example"
model_1_data = ia_maker.add_model(
  model_name='model_1',
  model_params={ 'input_shape': (2, 1) },
  load_model_params={},
  compile_params={
    'metrics': ['accuracy'],
    'optimizer': 'adam', 'loss': 'mse'
  },
)
```

```txt title=Return
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


```bash title=Output
GPU info: /physical_device:GPU:0

Model model_1/2022.07.18_10.04.03 added
```

:::info
The model defined won't be create immediately, this will be created when you call the `train` method. This in order to save your resources.
:::

## Update a model
The `update_model` method is used to update a model. Has the following parameters (all are optionals except `model_name`):

- **model_name** (string): Name of the model.
- **run_id** (string): Run id of the model.
- **model_params** (dictionary): to update the parameters to build the model.
- **compile_params** (dictionary): to update the parameters to compile the model.
- **load_model_params** (dictionary): to update the parameters to load the model.

```python title="Example"
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

```txt title=Output
Model model_1/modelo_dense_1 was updated
```

## Delete a model

The `delete_model` method is used to delete a model. Has the following parameters:

- **model_data** (string): Model data returned by `add_model` method.
- **delete_folder** (boolean): Delete the folder of the model, by default is `False`.

```python title="Example"
ia_maker.delete_model(model_1_data, delete_folder=True)
```

```txt title=Output
Model model_1/2022.07.18_10.04.03 was deleted
```

## Show models

The `show_models` method is used to show the models defined via standard output.

```python title="Example"
print('Show all models:')
ia_maker.show_models()
```

```bash title=Output
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
