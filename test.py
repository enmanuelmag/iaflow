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
  model_name='model_dense', # Name of the model
  models_folder='./models', # Path to folder where models will be saved
  run_id=None, # Optional run id to load model
  builder_function=custom_builder, # Function to build your or yours models
  model_params={ 'input_shape': (2, 1) }, # Parameters for builder function
  compile_params={ # Parameters for model compilation, see documentation tf.keras.models.compile
    'optimizer': 'adam', 'loss': 'mse',
    'metrics': ['accuracy']
  },
  load_model_params={} # Parameters for model loading, see documentation of tf.keras.models.load_model
)

print(
f"""
Model: {model} (the tf keras model)
Run id: {run_id} (this is a unique identifier for this run)
Model path: {path_model} (this is the path to the model folder)
Checkpoint path: {check_path} (this is the path to the model checkpoint)
Model ident: {model_ident} (this is a unique identifier for this model, include model name and parameters)
"""
)

all_data = tf.data.Dataset.from_tensor_slices((
  tf.random.uniform([1000, 2]),
  tf.random.uniform([1000, 1])
))
train_ds = all_data.take(5)
val_ds = all_data.skip(5)



mlw.train(
  model=model, # The tf keras model
  check_path=check_path, # Path to the model checkpoint
  path_model=path_model, # Path to the model folder
  train_ds=train_ds, # Training dataset, could be a tf.data.Dataset, a list of tensors or a tensor
  val_ds=val_ds, # Validation dataset, could be a tf.data.Dataset, a list of tensors or a tensor
  batch=32, # Batch size
  epochs=100, # Number of epochs
  initial_epoch=0, # Initial epoch
  checkpoint_params={ # Parameters for model checkpoint, see documentation tf.keras.callbacks.ModelCheckpoint
    'monitor': 'accuracy', 'verbose': 1,
    'save_best_only': False, 'save_freq':'epoch'
  },
  params_notifier={ # Parameters for notifier, see documentation https://pypi.org/project/notify-function/#description
    'title': 'Training update',
    'webhook_url': os.environ.get('WEBHOOK_URL'),
    'frequency_epoch': 20 # This will send a notification every 20 epochs, by default it is every epoch
  }
)
