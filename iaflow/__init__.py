import os
import gc
import json
import time
import shutil
import typing as T
import tensorflow as tf
import subprocess as sp

from datetime import datetime
from notifier import Notifier
from tensorflow.keras.backend import clear_session

SLIP_KEYS = [ 'run_id', 'model_name' ]

class NotifierCallback(tf.keras.callbacks.Callback):

  def __init__(self,
    title: str = 'Training update',
    email: str = None,
    chat_id: str = None,
    api_token: str = None,
    webhook_url: str = None,
    frequency_epoch: int = 1,
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
    self.frequency_epoch = frequency_epoch

  def on_epoch_end(self, batch, logs={}):
    self.epoch_count += 1
    if self.epoch_count % self.frequency_epoch != 0:
      return

    try:
      parsed_metrics = f'Epoch {self.epoch_count}\n'
      for key, value in logs.items():
        key = key.replace('_', ' ').capitalize()
        parsed_metrics += f'{key}: {value}\n'
      self.notifier(msg=parsed_metrics)
    except Exception as e:
      print('There was an error sending the notification:', e)

class ParamsNotifier(T.TypedDict):
  title: T.Optional[str]
  email: T.Optional[str]
  chat_id: T.Optional[str]
  api_token: T.Optional[str]
  webhook_url: T.Optional[str]

def NoImplementedError(message: str):
  raise NotImplementedError(message)


class IAFlow(object):

  def __init__(self,
    models_folder: str,
    builder_function: T.Callable = lambda **kwargs: NoImplementedError('Builder function not implemented'),
    batch: int = 32,
    epochs: int = 100,
    checkpoint_params: T.Dict = {},
    tensorboard_params: T.Dict = {},
    params_notifier: ParamsNotifier = None,
    callbacks: T.Union[T.List[T.Any], T.Any] = [],
    val_ds: T.Union[tf.data.Dataset, T.List[T.Any], T.Any] = None,
    train_ds: T.Union[tf.data.Dataset, T.List[T.Any], T.Any] = None,
  ):
    self.models = {}
    self.callbacks = callbacks
    self.models_folder = models_folder
    self.params_notifier = params_notifier
    self.builder_function = builder_function
    self.checkpoint_params = checkpoint_params
    self.tensorboard_params = tensorboard_params
    self.batch = batch
    self.epochs = epochs
    self.val_ds = val_ds
    self.train_ds = train_ds

    self.notifier = NotifierCallback(**params_notifier) if params_notifier else None

  def __delete_by_path(self, path: str, is_dir: bool = False):
    if is_dir:
      shutil.rmtree(path, ignore_errors=True)
    else:
      try:
        os.remove(path, ignore_errors=True)
      except:
        pass

  def __find_endwith(self, path: str, endwith: str):
    if not os.path.exists(path):
      os.makedirs(path, exist_ok=True)
      return None

    for filename in os.listdir(path):
      if filename.endswith(endwith):
        return os.path.join(path, filename)
    return None

  def __get_params_models(self, load_model: bool, path_model: str, model_params: T.Dict):
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

  def __create_file(self, path: str, content: T.Any, mode: str = 'w', is_json: bool = False):
    if not os.path.exists(path):
      with open(path, mode) as file:
        if is_json:
          json.dump(content, file)
        else:
          file.write(content)

  def set_builder_function(self, builder_function: T.Callable):
    self.builder_function = builder_function

  def get_model(self, model_name: str, run_id: str, force_creation: bool = True):
    
    model = self.models.get(model_name)
    if model is None:
      print(f'Model {model_name} not found')
      return None
    
    run_id_data = model.get(run_id)
    if run_id_data is None:
      print(f'Run {run_id} not found')
      return None

    check_path = run_id_data.get('check_path')
    if os.path.exists(check_path) and not force_creation:
      print(f'Loading model from {check_path}')
      model = tf.keras.models.load_model(**run_id_data.get('load_model_params'))
    else:
      print(f'Force creation is {force_creation}. Deleting old logs and model')
      self.__delete_by_path(run_id_data.get('log_dir'), is_dir=True)
      self.__delete_by_path(run_id_data.get('check_path'), is_dir=False)
      
      print('Creating model')
      model = self.builder_function(**run_id_data.get('model_params'))
      model.compile(**run_id_data.get('compile_params'))

    return model, run_id_data

  def delete_model(self, model_data: T.Dict, delete_folder: bool = False):
    model_name = model_data.get('model_name')
    run_id = model_data.get('run_id')
  
    model = self.models.get(model_name)
    if model is None:
      print(f'Model {model_name} not found')
      return None
    
    run_id_data = model.get(run_id)
    if run_id_data is None:
      print(f'Run {run_id} not found')
      return None

    if delete_folder:
      self.__delete_by_path(run_id_data.get('path_model'), is_dir=True)
    
    del model[run_id]
    if len(model) == 0:
      del self.models[model_name]
    
    print(f'Model {model_name}/{run_id} deleted')

  def show_models(self):
    print('Models:')
    for model_name, model in self.models.items():
      print(f'  {model_name}')
      for run_id, run_id_data in model.items():
        print(f'    {run_id}')
        for key, value in run_id_data.items():
          if key not in SLIP_KEYS:
            print(f'      {key}: {value}')

  def add_model(
    self,
    model_name: str,
    model_params: T.Dict = {},
    compile_params: T.Dict = {},
    load_model_params: T.Union[T.Dict, None] = {}
  ) -> T.Tuple[tf.keras.Model, str, str, str]:

    models_folder = self.models_folder
    model_params = self.__get_params_models(False, models_folder, model_params)
    model_params_str = '_'.join(map(str, model_params.values()))
    model_ident = '_'.join([ model_name, model_params_str ])

    
    run_id = datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
    path_model = os.path.join(models_folder, model_name, run_id)

    if not os.path.exists(path_model):
      os.makedirs(path_model)
    
    check_path = f'{path_model}/{model_ident}_checkpoint.h5'

    os.makedirs(path_model, exist_ok=True)
    path_params = f'{path_model}/{model_name}_params.json'
    self.__create_file(path_params, model_params, is_json=True)

    try:
      gpu_info = tf.config.experimental.list_physical_devices('GPU')
    except:
      gpu_info = ''

    info_env = gpu_info[0].name if len(gpu_info) > 0 else 'Not connected to a GPU'
    print(f'GPU info: {info_env}\n\n')

    model_data = self.models.get(model_name, {})
    
    load_model_params['filepath'] = check_path
    model_data[run_id] = {
      'run_id': run_id,
      'model_name': model_name,
      'model_ident': model_ident,
      'model_params': model_params,
      'compile_params': compile_params,
      'load_model_params': load_model_params,
      'check_path': check_path,
      'path_model': path_model,
      'log_dir': f'{path_model}/logs',
    }

    self.models[model_name] = model_data
    return model_data[run_id]

  def train(
    self,
    model_data: T.Dict,
    train_ds: T.Union[tf.data.Dataset, T.List[T.Any], T.Any] = None,
    val_ds: T.Union[tf.data.Dataset, T.List[T.Any], T.Any] = None,
    force_creation: bool = False,
    batch: int = 32,
    epochs: int = 100,
    initial_epoch: int = 0
  ):
    batch = batch or self.batch
    epochs = epochs or self.epochs
    val_ds = val_ds or self.val_ds
    train_ds = train_ds or self.train_ds

    model_name = model_data.get('model_name')
    run_id = model_data.get('run_id')

    self.clear_session()
    model, run_data = self.get_model(model_name, run_id, force_creation)

    if self.notifier:
      self.callbacks.append(self.notifier)

    self.checkpoint_params['filepath'] = run_data.get('check_path')
    self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(**self.checkpoint_params))

    self.tensorboard_params['log_dir'] = run_data.get('log_dir')
    self.callbacks.append(tf.keras.callbacks.TensorBoard(**self.tensorboard_params))

    use_tf_dataset = False
    if isinstance(train_ds, tf.data.Dataset) and isinstance(val_ds, tf.data.Dataset):
      use_tf_dataset = True
      train_ds = train_ds.batch(batch)
      val_ds = val_ds.batch(batch)

    start_time = time.time()
    model.fit(
      train_ds,
      epochs=epochs,
      callbacks=self.callbacks,
      validation_data=val_ds,
      initial_epoch=initial_epoch,
      batch_size=batch if not use_tf_dataset else None,
    )
    print(f'Training time: {time.time() - start_time}')

    self.clear_session()
    del model

  def save(self, path: str):
    data = self.models
    path = path or self.models_folder

    with open(f'{path}/models.json', 'w') as file:
      json.dump(data, file)
    print(f'Saved data to {path}/models.json')

  def clear_session(self):
    clear_session()
    gc.collect()

