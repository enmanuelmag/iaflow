import os
import json
import tensorflow as tf
from iaflow import IAFlow


def custom_builder(input_shape):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Input(input_shape),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
  ])
  return model

params_notifier = { # Parameters for notifier, see documentation https://pypi.org/project/notify-function/#description
  'title': 'Training update',
  'webhook_url': os.environ.get('WEBHOOK_URL'),
  'frequency_epoch': 20 # This will send a notification every 20 epochs, by default it is every epoch
}

all_data = tf.data.Dataset.from_tensor_slices((
  tf.random.uniform([1000, 2]),
  tf.random.uniform([1000, 1])
))
train_ds = all_data.take(5)
val_ds = all_data.skip(5)

# Creation of the IAFlow instance
ia_maker = IAFlow(
  val_ds=val_ds,
  train_ds=train_ds,
  models_folder='./models',
  params_notifier=params_notifier,
  builder_function=custom_builder
)

model_1_data = ia_maker.add_model(
  model_name='model_1',
  model_params={ 'input_shape': (2, 1) }, # Parameters for builder function
  load_model_params={}, # Parameters for model loading, see documentation of tf.keras.models.load_model
  compile_params={ # Parameters for model compilation, see documentation tf.keras.models.compile
    'optimizer': 'adam', 'loss': 'mse',
    'metrics': ['accuracy']
  },
)
print(f'Data of model_1: {json.dumps(model_1_data, indent=4)}')

print('Show all models:')
ia_maker.show_models()

ia_maker.train(model_1_data, batch=32, epochs=5)

ia_maker.clear_session()

print('Lading model trained')
ia_maker.train(model_1_data, batch=32, epochs=5)

ia_maker.delete_model(model_1_data, delete_folder=True)
