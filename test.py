import ml_workflow as mlw
import os

import tensorflow as tf


def custom_builder(input_shape):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Input(input_shape),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
  ])
  return model

model, model_ident, check_path, path_model = mlw.build(
  load_model=True,
  model_name='model_test',
  path_model='./test',
  #model_params={'input_shape': (2, 1) },
  compile_params={
    'optimizer': 'adam', 'loss': 'mse',
    'metrics': ['accuracy']
  },
  builder_function=custom_builder,
)

#create a dataset where x 1D vecto ans y scalar
all_data = tf.data.Dataset.from_tensor_slices((
  tf.random.uniform([1000, 2]),
  tf.random.uniform([1000, 1])
))
train_ds = all_data.take(5)
val_ds = all_data.skip(5)

mlw.train(
  model, check_path, path_model,
  train_ds, val_ds, batch=32, epochs=2,
  checkpoint_params={ 'monitor': 'accuracy', 'verbose': 1, 'save_best_only': False },
)
