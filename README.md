# IA Workflow
This library help to create models with identifiers, checkpoints, logs and metadata automatically, in order to make the training process more efficient and traceable.

## Requirements

- Tensorflow 2.5 or higher.
- [notify-function](https://pypi.org/project/notify-function/#description) 1.5.0 or higher.

## Usage

This library has a maser class called `IAFlow`, that has function, the most important are:
 - `add_model`: to add a new model to the internal structure of maker models to build later to training.
 - `train`: to train a model with a internal callbacks to save metrics with Tensorboard, save the model with checkpoints and send notification to discord channel, email or telegram (all this is optional).

### Constructor

```python
from iaflow import IAFlow

def custom_builder(input_shape):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Input(input_shape),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
  ])
  return model

ia_maker = IAFlow(
  val_ds=val_ds, # Dataset to validate the model, you can change this when call `train` method
  train_ds=train_ds, # Dataset to train the model, you can change this when call `train` method
  models_folder='./models', # Folder to save the models
  params_notifier=params_notifier, # Notifier to send notification to discord channel, email or telegram (all this is optional)
  builder_function=custom_builder # Function to build the model, you can change this when call `train` method
)
```

### `add_model` method
The build method is used to create a model in a defined structure of folder. As below:

```
`models_folder`
└─ `model_name`
   └─ `run_id`
      ├─ logs                               
      │  ├─ train
      │  ├─ validation
      ├─ `model_name`_`model_params`_checkpoint.h5
      └─ `model_name`_params.json
```

```python
model_1_data = ia_maker.add_model(
  model_name='model_1', # Name of the model
  model_params={ 'input_shape': (2, 1) }, # Parameters for builder function
  load_model_params={}, # Parameters for model loading, see documentation of tf.keras.models.load_model
  compile_params={ # Parameters for model compilation, see documentation tf.keras.Models.compile
    'optimizer': 'adam', 'loss': 'mse',
    'metrics': ['accuracy']
  },
)
```

If you want to load a trained model you must send the parameter `run_id`, and you can skip the parameters like: `builder_function`, `model_params` and `compile_params`.

> Note: If your model use custom SubClass for custom Layers you must send the parameter `load_model_params` with the parameter `custom_objects` with the custom class. see documentation of tf.keras.models.load_model

#### Return
This method return a dictionary with information of the model


### `train` method

The train method is used to train a model.

#### Parameters

```python
ia_maker.train(model_1_data, batch=32, epochs=5)
```

If you loaded a trained model you should send a dictionary with key `model_name` and `run_id` that belong to the model desired to load and the initial epoch. Don't change the paths like `check_path` and `path_model`, because they are already setted properly when you loaded the model with the `build` method. Also don't send the key `filepath` on the `checkpoint_params` because it is already setted properly. inside the method.

If you want use `notify-function` lib you must send the parameter `params_notifier`. To see how to use and what do this library you can see the documentation of [notify-function](https://pypi.org/project/notify-function/#description). The message will have the epoch number and the values of each metric that you defined for your model.


<img src="assets/message.png" alt="Message example" width="300"/>


> If you use `notify-function` and you specify email maybe this will add a delay to the training process. To avoid this you can use another methods more faster like discord webhooks or telegram message or instead use the key `frequency_epoch` on the `params_notifier` reduce the rate of notifications.

### `show_models` method
This function show all models that you have added to the IAFlow.

### `delete_model` method
This function delete a model from the IAFlow. You must send a dictionary with key `model_name` and `run_id` that belong to the model desired to delete. If you send the parameter `delete_folder` it will delete the folder of the model.

## FAQs

If you have any question find a bug or feedback, please contact me with a email to [enmanuelmag@cardor.dev](mailto:enmanuelmag@cardor.dev)

Made with ❤️ by [Enmanuel Magallanes](https://cardor.dev)
