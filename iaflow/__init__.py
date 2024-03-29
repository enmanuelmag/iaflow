import os
import gc
import json
import time
import copy
import shutil
import numpy as np
import pickle as pkl
import seaborn as sns
import tensorflow as tf
import subprocess as sp
import matplotlib.pyplot as plt

from datetime import datetime
from notifier import Notifier
from tensorflow.keras.backend import clear_session

__SLIP_KEYS__ = [ 'run_id', 'model_name' ]

__METRIC_CONFIG__ = {
  "name": "val_loss",
  "initial": float('inf'),
  "operator": "min",
  "monitor": True
}

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
    if self.epoch_count % self.frequency_epoch != 0 or self.epoch_count == 0:
      return

    try:
      parsed_metrics = f'Epoch {self.epoch_count}\n'
      for key, value in logs.items():
        key = key.replace('_', ' ').capitalize()
        parsed_metrics += f'{key}: {value}\n'
      self.notifier(msg=parsed_metrics)
    except Exception as e:
      print('There was an error sending the notification:', e)

def NoImplementedError(message: str):
  raise NotImplementedError(message)


class IAFlow(object):

  def __init__(
    self,
    models_folder: str,
    builder_function = lambda **kwargs: NoImplementedError('Builder function not implemented'),
    callbacks = [],
    checkpoint_params = {},
    tensorboard_params = {},
    params_notifier = None,
  ):
    self.models = {}
    self.datasets = {}
    self.callbacks = callbacks
    self.models_folder = models_folder
    self.params_notifier = params_notifier
    self.builder_function = builder_function
    self.checkpoint_params = checkpoint_params
    self.tensorboard_params = tensorboard_params

    self.notifier = NotifierCallback(**params_notifier) if params_notifier else None

    try:
      gpu_info = tf.config.experimental.list_physical_devices('GPU')
    except:
      gpu_info = ''

    info_env = f'GPU connected: {gpu_info}' if len(gpu_info) > 0 else 'Not connected to a GPU'
    print(f'GPU info: {info_env}')

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

  def __get_params_models(self, load_model: bool, path_model: str, model_params):
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

  def __create_file(self, path: str, content, mode: str = 'w', is_json: bool = False):
    if not os.path.exists(path):
      with open(path, mode) as file:
        if is_json:
          json.dump(content, file)
        else:
          file.write(content)

  def __get_config(self):
    pass

  def set_builder_function(self, builder_function):
    self.builder_function = builder_function
  
  def set_notifier_parameters(self, params):
    self.params_notifier = params
    self.notifier = NotifierCallback(**params)

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
      params = {
        **run_id_data.get('load_model_params', {}),
        'filepath': check_path,
      }
      model = tf.keras.models.load_model(**params)
      return model, run_id_data
    elif force_creation:
      print(f'Force creation is {force_creation}. Deleting old logs and model')
      self.__delete_by_path(run_id_data.get('log_dir'), is_dir=True)
      self.__delete_by_path(run_id_data.get('check_path'), is_dir=False)

      print('Creating model')
      model = self.builder_function(**run_id_data.get('model_params'))
      model.compile(**run_id_data.get('compile_params'))

    return model, run_id_data

  def show_models(self):
    print('Models:')
    for model_name, model in self.models.items():
      print(f'  {model_name}')
      for run_id, run_id_data in model.items():
        print(f'    {run_id}')
        for key, value in run_id_data.items():
          if key not in __SLIP_KEYS__:
            print(f'      {key}: {value}')

  def add_dataset(
    self,
    name: str,
    epochs: int,
    train_ds,
    batch_size: int = None,
    shuffle_buffer: int = None,
    val_ds = None,
    test_ds = None
  ):
    if name in self.datasets:
      print(f'Dataset {name} already exists')

    self.datasets[name] = {
      'train_ds': train_ds,
      'epochs': epochs,
      'batch_size': batch_size,
      'shuffle_buffer': shuffle_buffer,
    }
    if val_ds is not None:
      self.datasets[name]['val_ds'] = val_ds
    if test_ds is not None:
      self.datasets[name]['test_ds'] = test_ds

    print(f'Dataset {name} was added')

  def update_dataset(
    self,
    name: str = None,
    epochs: int = None,
    batch_size: int = None,
    shuffle_buffer: int = None,
    train_ds = None,
    val_ds = None,
    test_ds = None
  ):
    if name not in self.datasets:
      print(f'Dataset {name} not found')

    if batch_size is not None:
      self.datasets[name]['batch_size'] = batch_size
    if shuffle_buffer is not None:
      self.datasets[name]['shuffle_buffer'] = shuffle_buffer
    if epochs is not None:
      self.datasets[name]['epochs'] = epochs
    if train_ds is not None:
      self.datasets[name]['train_ds'] = train_ds
    if val_ds is not None:
      self.datasets[name]['val_ds'] = val_ds
    if test_ds is not None:
      self.datasets[name]['test_ds'] = test_ds

    print(f'Dataset {name} was updated')

  def delete_dataset(self, name: str):
    if name not in self.datasets:
      print(f'Dataset {name} not found')

    del self.datasets[name]
    print(f'Dataset {name} was deleted')

  def add_model(
    self,
    model_name: str,
    run_id: str = None,
    model_params = {},
    compile_params = {},
    load_model_params = {}
  ):

    models_folder = self.models_folder
    model_params_str = '_'.join(map(str, model_params.values()))
    model_ident = '_'.join([ model_name, model_params_str ])

    runs_model = self.models.get(model_name, {})
    if run_id is not None and run_id in runs_model:
      raise ValueError(f'Model {model_name}/{run_id} already exists')

    run_id = run_id or datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
    path_model = os.path.join(models_folder, model_name, run_id)

    if not os.path.exists(path_model):
      os.makedirs(path_model)
    
    check_path = f'{path_model}/{model_ident}_checkpoint.h5'

    os.makedirs(path_model, exist_ok=True)
    path_params = f'{path_model}/{model_name}_params.json'
    self.__create_file(path_params, model_params, is_json=True)

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
    #self.__save()
    print(f'Model {model_name}/{run_id} was added')
    return model_data[run_id]

  def update_model(
    self,
    model_name: str,
    run_id: str,
    model_params = {},
    compile_params = {},
    load_model_params = {}
  ):

    models_folder = self.models_folder
    model_params_str = '_'.join(map(str, model_params.values()))
    model_ident = '_'.join([ model_name, model_params_str ])

    model_data = self.models.get(model_name, {})
    if run_id not in model_data:
      raise ValueError(f'Model {model_name}/{run_id} not found')

    model = runs_model[run_id]

    if model_params is not None:
      model['model_params'] = model_params
    if compile_params is not None:
      model['compile_params'] = compile_params
    if load_model_params is not None:
      model['load_model_params'] = load_model_params

    model_data[run_id] = model
    self.models[model_name] = model_data
    print(f'Model {model_name}/{run_id} was updated')
    return model_data[run_id]

  def delete_model(self, model_data, delete_folder: bool = False):
    model_name = model_data.get('model_name')
    run_id = model_data.get('run_id')
  
    model = self.models.get(model_name)
    if model is None:
      print(f'Model {model_name} not found')
    
    run_id_data = model.get(run_id)
    if run_id_data is None:
      print(f'Run {run_id} not found')

    if delete_folder:
      self.__delete_by_path(run_id_data.get('path_model'), is_dir=True)
    
    del model[run_id]
    if len(model) == 0:
      del self.models[model_name]

    #self.__save()
    print(f'Model {model_name}/{run_id} was deleted')

  def train(
    self,
    model_data,
    dataset_name: str,
    batch_size: int = None,
    initial_epoch: int = 0,
    shuffle_buffer: int = None,
    force_creation: bool = False,
    epochs = None,
    train_ds = None,
    val_ds = None,
    metric_config = None,
    monitor = None,
    group_plots = None,
  ):
    epochs = epochs or self.datasets.get(dataset_name, {}).get('epochs', 100)
    batch_size = batch_size or self.datasets.get(dataset_name, {}).get('batch_size', None)
    shuffle_buffer = shuffle_buffer or self.datasets.get(dataset_name, {}).get('shuffle_buffer', None)

    val_ds = val_ds or self.datasets.get(dataset_name, {}).get('val_ds')
    train_ds = train_ds or self.datasets.get(dataset_name, {}).get('train_ds')

    if train_ds is None or val_ds is None:
      raise ValueError('train_ds and val_ds must be provided on the instance creation or train method call')

    model_name = model_data.get('model_name')
    run_id = model_data.get('run_id')

    self.clear_session()
    print(f'Training {model_name}/{run_id}...')
    print(f'Epochs: {epochs}')
    print(f'Batch size: {batch_size}')
    print(f'Shuffle buffer: {shuffle_buffer}')
    print(f'Start time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    
    model, run_data = self.get_model(model_name, run_id, force_creation)

    if self.notifier:
      self.callbacks.append(self.notifier)

    self.checkpoint_params['filepath'] = run_data.get('check_path')
    self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(**self.checkpoint_params))

    self.tensorboard_params['log_dir'] = run_data.get('log_dir')
    self.callbacks.append(tf.keras.callbacks.TensorBoard(**self.tensorboard_params))

    use_tf_dataset = isinstance(train_ds, tf.data.Dataset) and isinstance(val_ds, tf.data.Dataset)
    if batch_size is not None and isinstance(train_ds, tf.data.Dataset) and isinstance(val_ds, tf.data.Dataset):
      use_tf_dataset = True
      train_ds = train_ds.batch(batch_size)
      val_ds = val_ds.batch(batch_size)

    if shuffle_buffer is not None and use_tf_dataset:
      train_ds = train_ds.shuffle(shuffle_buffer)
      val_ds = val_ds.shuffle(shuffle_buffer)

    start_time = time.time()

    history = model.fit(
      train_ds,
      epochs=epochs,
      callbacks=self.callbacks,
      validation_data=val_ds,
      initial_epoch=initial_epoch,
      batch_size=batch_size if not use_tf_dataset else None,
    )
    print(f'Training time: {time.time() - start_time}')
    self.clear_session()

    path_save = run_data.get('path_model', '.')
    df_history = parse_history(history.history, path_save)
    plot_history(df_history, path_save, metric_config, monitor, group_plots)

    return history

  def __save(self):
    """
    Future implementation
    """
    return
    path = f'{self.models_folder}/data.iaflow'

    with open(path, 'wb') as file:
      print(self.models_folder)
      pkl.dump(self, file, protocol=pkl.HIGHEST_PROTOCOL)

    print(f'Saved data to {path}')

  def clear_session(self):
    clear_session()
    gc.collect()

  def __evaluate_metric(self, current_value = 0, best_value = 0, operator = ''):
    if operator == 'min':
      return current_value < best_value
    elif operator == 'max':
      return current_value > best_value
    else:
      raise ValueError(f'Operator {operator} not supported')

  def parse_history(self, history, path):
    df_history = pd.DataFrame()
    for key in history.keys():
      df_history[key.capitalize()] = history[key]

    df_history.to_csv(f'{path}/history.csv', index=False)
    return df_history

  def plot_history(self, df_history, path, metric_config=__METRIC_CONFIG__, monitor='val_loss', group_plots=[['loss', 'val_loss'], ['accuracy', 'val_accuracy']]):
    assert df_history is not None, 'History must be provided'
    assert path is not None, 'Path must be provided'
    assert monitor in df_history.columns, f'Monitor {monitor} not found in history'
  
    keys_columns = df_history.columns
    monitor_data = df_history[monitor].values

    best_value = metric_config["initial"]
    assert best_value is not None, 'Initial value must be provided'
    
    operator = metric_config["operator"]
    assert operator in ['min', 'max'], f'Operator {operator} not supported'

    results = {}
    for idx, value in enumerate(monitor_data):
      if self.__evaluate_metric(value, best_value, operator):
        best_value = value
        for key in keys_columns:
          results[key] = df_history[key].values[idx]

    print('Best results:')
    with open(f'{path}/best_results.txt', 'w') as f:
      for key, value in results.items():
        line = f'{key}: {value}'
        f.write(f'{line}\n')
        print(line)

    for group in group_plots:
      df_group = pd.DataFrame()

      for key in group:
        assert key in keys_columns, f'Key {key} not found in history'
        df_group[key] = df_history[key].values

      sns.lineplot(data=df_group, x='Epoch', y='Value')
      plt.title(f'{" vs ".join(df_group.columns)}')
      fig.tight_layout()
      plt.show()

      if path is not None:
        fig.savefig(f'{path}/{"_".join(df_group.columns)}.png', dpi=300)
