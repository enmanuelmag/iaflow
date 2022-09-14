---
sidebar_position: 1
---

# Main class

This library has a maser class called `IAFlow`, that has functions to management model creation and training. In this section, we will see the CRUD for models.

## Constructor

The constructor of `IAFlow` class has the following parameters:

- **models_folder**: Folder to save the models.
- **params_notifier**: Parameters for notifier, see documentation [here](https://pypi.org/project/notify-function/#description).
- **builder_function**: Function to build the model.
- **checkpoint_params** (optional): Parameters for checkpoint, see documentation [here](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint).
- **tensorboard_params** (optional): Parameters for tensorboard, see documentation [here](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard).

```python title="IAFlow constructor"
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
  builder_function=custom_builder,
  checkpoint_params={
    'monitor': 'val_loss',
    'save_best_only': True,
    'save_weights_only': True
  },
  tensorboard_params={
    'histogram_freq': 1,
    'write_graph': True,
    'write_images': True
  }
)
```

## Update builder function

The `update_builder_function` method is used to update the builder function. Has the following parameters:

- **builder_function**: New function to build the model.

```python title=Example
ia_maker.update_builder_function(custom_builder_2)
```

## Update notifier parameters

The `update_notifier_parameters` method is used to update the notifier parameters. Has the following parameters:

- **params_notifier**: New parameters for notifier, see documentation [here](https://pypi.org/project/notify-function/#description).

```python title=Example
ia_maker.update_notifier_parameters(params_notifier_2)
```