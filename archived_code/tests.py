import tensorflow_datasets as tfds
import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST
import jax.numpy as jnp
from jax import vmap, grad
import pandas as pd

import equinox as eqx
import optax
import jax.random as jrandom
import time
import pickle
import os
import argparse

import torch

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)


batch_size = 128 # 256
LABEL = 'CO'

def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

class MyDataset(data.Dataset):
    def __init__(self, df, target=LABEL, mean=0, std=1, seed=0):
        df = df.sample(frac=1.0, random_state=seed)
        df = df.fillna(df.median())
        df = (df - mean) / std
        self.labels = df[target].to_numpy()
        self.inputs = df.drop(columns=[target]).to_numpy()
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


commands = tfds.load('speech_commands')
train_set = commands['train']
test_set = commands['test']

train_set = train_set.batch(batch_size)

for item in tfds.as_numpy(train_set):
    pass
