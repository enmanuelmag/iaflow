import os
import tensorflow as tf
import ml_workflow as mlw


def custom_builder(input_shape):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Input(input_shape),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
  ])
  return model


model, run_id, model_ident, check_path, path_model = mlw.build(
  models_folder='./models',
  model_name='model_dense',
  run_id=None, #'2022.07.13_10.32.23',
  builder_function=custom_builder,
  model_params={ 'input_shape': (2, 1) },
  compile_params={
    'optimizer': 'adam', 'loss': 'mse',
    'metrics': ['accuracy']
  }
)

print(f'Run id: {run_id}, model ident: {model_ident}, check path: {check_path}, path model: {path_model}')

all_data = tf.data.Dataset.from_tensor_slices((
  tf.random.uniform([1000, 2]),
  tf.random.uniform([1000, 1])
))
train_ds = all_data.take(5)
val_ds = all_data.skip(5)



mlw.train(
  model, check_path, path_model, train_ds, val_ds,
  batch=32, epochs=100, initial_epoch=0,
  checkpoint_params={ 'monitor': 'accuracy', 'verbose': 1, 'save_best_only': False, 'save_freq':'epoch' },
  params_notifier={
    'title': 'Training update',
    'webhook_url': 'https://ptb.discord.com/api/webhooks/996699042266492939/xcshnimJah-Uds6tY6BKh_E5e5OZ_6tUYgxnABrdVH8LyXJE3XNgfZ0OTQLkWMqHuud9',
    'frequency_epoch': 20
  }
)
