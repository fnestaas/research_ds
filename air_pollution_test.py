import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST
import jax.numpy as jnp
from jax import vmap, grad
import pandas as pd

from models import NeuralODEClassifier as node
from models.Func import Func
from models.NeuralODE import StatTracker

import equinox as eqx
import optax
import jax.random as jrandom
import time
import pickle
import os
import argparse

import torch

# parser = argparse.ArgumentParser('Run MNIST test')
# parser.add_argument('FUNC', ) # RegularFunc or whatever else
# parser.add_argument('SKEW', type=str)
# parser.add_argument('SEED', type=str)
# parser.add_argument('dst')

# args = parser.parse_args()

# FUNC = args.FUNC
# SEED = int(args.SEED)
# SKEW = args.SKEW == 'True' # This was wrong when I ran the tests...
# dst = args.dst
# print(f'\nrunning with args {args}\n')

FUNC = 'RegularFunc'
SKEW = False
SEED = 0
dst = f'tests/pollution_{FUNC=}_{SKEW=}{SEED}'

print(dst)

LABEL = 'CO'

np.random.seed(SEED)
torch.manual_seed(SEED)

TRACK_STATS = True 
DO_BACKWARD = True

batch_size = 128 # 256

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


dir = 'data/air_quality/air_quality/'
files = os.listdir(dir)

df = pd.read_csv(dir + str(files[0]))
df = df.select_dtypes(include=['float64', 'int64'])# df.loc[:, df.dtypes != object]
means = df.mean()
stds = df.std()

msk = np.random.rand(len(df)) < 0.8

test_set = df[msk]
train_set = df[~msk]

# Get full test dataset
dataset_train = MyDataset(train_set, mean=means, std=stds, seed=SEED)
training_generator = NumpyLoader(dataset_train, batch_size=batch_size, num_workers=0)
dataset_test = MyDataset(test_set, mean=means, std=stds, seed=SEED)
testing_generator = NumpyLoader(dataset_test, batch_size=batch_size, num_workers=0)

test_set_ = test_set.sample(frac=.01, random_state=SEED)
test_input = test_set_.drop(columns=[LABEL]).to_numpy()
test_output = test_set_[LABEL].to_numpy()

def main(
    lr=1e-3, 
    n_epochs=4,
    steps_per_epoch=200,
    seed=SEED,
    print_every=10,
):
    key = jrandom.PRNGKey(seed)
    _, model_key, l = jrandom.split(key, 3)

    d = 10
    depth = 3
    width_size = 64
    if FUNC == 'PDEFunc':
        func = node.PDEFunc(d=d, width_size=width_size, depth=depth, integrate=False, skew=SKEW, seed=seed) # number of steps taken to solve is very important. Use more advanced method?
    elif FUNC == 'RegularFunc':
        func = node.RegularFunc(d=d, width_size=width_size, depth=depth, seed=seed,)
        # model = node.NeuralODEClassifier(func, in_size=28*28, out_size=10, key=model_key, use_out=True)
    else:
        raise NotImplementedError
    model = node.NeuralODEClassifier(func, in_size=15, out_size=1, key=model_key, use_out=True, activation=lambda x: x)
    grad_tracker = StatTracker(['adjoint_norm'])

    validation_loss = []

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi, labels):
        y_pred = vmap(model, in_axes=(None, 0, None))(ti, yi, TRACK_STATS)
        return _loss_func(labels, y_pred, model, lam=1e0)

    # @eqx.filter_jit
    def make_step(ti, yi, model, opt_state, labels):
        loss, grads = grad_loss(model, ti, yi, labels)
        updates, opt_state = optim.update(grads, opt_state) 

        if DO_BACKWARD: 
            backward_pass = model.backward(ti, yi, _loss_func, labels)
            n_adj = backward_pass.ys.shape[-1] // 2
            adjoint = backward_pass.ys[:, :, :n_adj]
            adjoint_norm = jnp.linalg.norm(adjoint, axis=-1)
            if TRACK_STATS:
                grad_tracker.update({'adjoint_norm': adjoint_norm})

        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    def _loss_func(y, y_pred, model, lam=1e0):
        loss = jnp.square(y - y_pred.reshape((-1, ))) 
        return jnp.mean(loss) # + lam * jnp.mean(jnp.square(model.get_params()))

    for epoch in range(n_epochs):
        optim = optax.adabelief(lr)
        # optim = optax.adam(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        steps = steps_per_epoch
        print(f'\nepoch {epoch+1}/{n_epochs}')
        for step, (yi, labels) in zip( 
            range(steps), training_generator 
        ):
            #length = .5 + .5*int(epoch > 1)
            length = 1
            # _ts = jnp.array([0., length])
            _ts = jnp.linspace(0., length, 100)
            start = time.time()
            loss, model, opt_state = make_step(_ts, yi, model, opt_state, labels)
            end = time.time()
            if (step % print_every) == 0 or step == steps - 1:
                print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")
                nfe = jnp.mean(model.get_stats()['num_steps'][-1]) # goes up if we save more often!
                print(f'mean nfe: {nfe}')
                
                for step_, (test_input, test_output) in zip( 
                    range(1), training_generator 
                ):
                    preds = vmap(model, in_axes=(None, 0, None))(_ts, test_input, False)
                    acc = _loss_func(test_output, preds, model)
                    print(f'Test loss: {acc}') 
                    validation_loss.append(acc)
    return model, grad_tracker, validation_loss

model, grad_tracker, validation_loss = main()

def make_np(arr_list):
    try:
        res = [item.val._value for item in arr_list]
    except AttributeError:
        try:
            res = [item._value for item in arr_list]
        except AttributeError:
            res = [item.val.aval.val._value for item in arr_list]
    return res

if not os.path.exists(dst):
    os.makedirs(dst)

def save_jnp(to_save, handle):
    pickle.dump(make_np(to_save), handle)

# Save the model parameters
with open(dst + '/acc.pkl', 'wb') as handle:
    save_jnp(validation_loss, handle)

with open(dst + '/last_state.pkl', 'wb') as handle:
    save_jnp(model.get_params(), handle)

# Save stats
if TRACK_STATS:
    with open(dst + '/num_steps.pkl', 'wb') as handle:
        to_save = model.get_stats()['num_steps']
        save_jnp(to_save, handle)
        
    with open(dst + '/state_norm.pkl', 'wb') as handle:
        to_save = model.get_stats()['state_norm']
        save_jnp(to_save, handle)

    # with open(dst + '/grad_init.pkl', 'wb') as handle:
    #     to_save = model.get_stats()['grad_init']
    #     save_jnp(to_save, handle)

    with open( dst + '/adjoint_norm.pkl', 'wb') as handle:
        to_save = grad_tracker.attributes['adjoint_norm']
        save_jnp(to_save, handle)
