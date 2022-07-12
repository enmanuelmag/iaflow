import os
import time
import json
import typing as T
import tensorflow as tf
import subprocess as sp

from datetime import datetime
from notifier import Notifier


def __find_endwith(path: str, endwith: str):
  for filename in os.listdir(path):
    if filename.endswith(endwith):
      return os.path.join(path, filename)
  return None

def __get_params_models(load_model: bool, path_model: str, model_params: T.Dict):
  if not load_model:
    return model_params
  
  params = None
  path_params = __find_endwith(path_model, '.json')
  if path_params is not None:
    with open(path_params, 'r') as file_params:
      params = json.load(file_params)
  
  if params:
    return params

  return model_params

def __create_file(path: str, content: T.Any, mode: str = 'w', is_json: bool = False):
  if not os.path.exists(path):
    with open(path, mode) as file:
      if is_json:
        json.dump(content, file)
      else:
        file.write(content)


def build(
  model_name: str,
  path_model: str,
  builder_function: T.Callable,
  load_model: bool = False,
  model_params: T.Dict = {},
  compile_params: T.Dict = {},
  custom_objects: T.Union[T.Dict, None] = {}
) -> T.Tuple[tf.keras.Model, str, str, str]:

  model_params = __get_params_models(load_model, path_model, model_params)
  model_params_str = '_'.join(map(str, model_params.values()))
  model_ident = '_'.join([ model_name, model_params_str ])
  
  path_model = os.path.join(path_model, model_name)
  if not os.path.exists(path_model):
    os.makedirs(path_model)
  
  if load_model:
    check_path = __find_endwith(path_model, '.h5')
  else:
    check_path = f'{path_model}/{model_ident}_checkpoint.h5'

  os.makedirs(path_model, exist_ok=True)

  path_params = f'{path_model}/{model_name}_params.json'
  
  __create_file(path_params, model_params, is_json=True)

  if os.path.isfile(check_path) and load_model:
    found_model = True
    print(f'Model found, building saved model at {check_path}')
  else:
    found_model = False
    print('Creating new model')

  try:
    gpu_info = tf.config.experimental.list_physical_devices('GPU')
  except:
    gpu_info = ''

  info_env = gpu_info[0].name if len(gpu_info) > 0 else 'Not connected to a GPU'
  print(f'\nGPU info: {info_env}\n')

  if found_model:
    model = tf.keras.models.load_model(check_path, custom_objects=custom_objects, compile=True)
  else:
    model = builder_function(**model_params)
    model.compile(**compile_params)

  return model, model_ident, check_path, path_model


class NotifierCallback(tf.keras.callbacks.Callback):

  def __init__(self,
    title: str = 'Training update',
    email: str = None,
    chat_id: str = None,
    api_token: str = None,
    webhook_url: str = None,
  ):
    super().__init__()
    self.notifier = Notifier(
      title=title,
      email=email,
      chat_id=chat_id,
      api_token=api_token,
      webhook_url=webhook_url
    )
    self.epoch_count = 0

  def on_epoch_end(self, batch, logs={}):
    self.epoch_count += 1
    try:
      parsed_metrics = f'Epoch {self.epoch_count}\n'
      for key, value in logs.items():
        key = key.replace('_', ' ').capitalize()
        parsed_metrics += f'{key}: {value}\n'
      self.notifier(msg=parsed_metrics)
    except Exception as e:
      print('There was an error sending the notification:', e)


class __ParamsNotifier(T.TypedDict):
  title: T.Optional[str]
  email: T.Optional[str]
  chat_id: T.Optional[str]
  api_token: T.Optional[str]
  webhook_url: T.Optional[str]

def train(
  model: tf.keras.Model,
  check_path: str,
  path_model: str,
  train_ds: T.Union[tf.data.Dataset, T.List[T.Any], T.Any],
  val_ds: T.Union[tf.data.Dataset, T.List[T.Any], T.Any],
  batch: int = 32,
  epochs: int = 100,
  initial_epoch: int = 0,
  callbacks: T.Union[T.List[T.Any], T.Any] = [],
  params_notifier: __ParamsNotifier = None,
  checkpoint_params: T.Dict = {},
  tensorboard_params: T.Dict = {},
):
  assert isinstance(train_ds, tf.data.Dataset) and isinstance(val_ds, tf.data.Dataset)
  
  if params_notifier:
    callbacks.append(NotifierCallback(**params_notifier))

  callbacks.append(tf.keras.callbacks.ModelCheckpoint(check_path, **checkpoint_params))
  callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=f'{path_model}/logs', **tensorboard_params))

  use_tf_dataset = False
  if isinstance(train_ds, tf.data.Dataset) and isinstance(val_ds, tf.data.Dataset):
    use_tf_dataset = True
    train_ds = train_ds.batch(batch)
    val_ds = val_ds.batch(batch)

  start_time = time.time()
  model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
    initial_epoch=initial_epoch,
    batch_size=batch if not use_tf_dataset else None,
  )

  print(f'Training time: {time.time() - start_time}')
