# This file is taken from
# https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html

# TODO: other datasets here: http://proceedings.mlr.press/v80/helfrich18a/helfrich18a.pdf
# https://github.com/ajithcodesit/lstm_copy_task/blob/master/LSTM_copy_task.py 

from turtle import backward
import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST
import jax.numpy as jnp
from jax import vmap, grad

from models import NeuralODEClassifier as node
from models.NeuralODE import StatTracker

import equinox as eqx
import optax
import jax.random as jrandom
import time
import pickle
import os

TRACK_STATS = True 
DO_BACKWARD = True
SKEW = True 
skew = 'skew' if SKEW else 'none'
dst = f'tests/mnist_run_{skew}'

batch_size = 256
n_targets = 10

def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

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

class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))

mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=FlattenAndCast())
training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)

train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)

# Get full test dataset
mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)

def main(
    lr_strategy=(3e-3, 3e-3),
    steps_strategy=(200, 200),
    length_strategy=(1, 1),
    seed=5678,
    print_every=100,
):
    key = jrandom.PRNGKey(seed)
    data_key, model_key, loader_key = jrandom.split(key, 3)

    # ts, ys = get_data(dataset_size, key=data_key) 
    # _, length_size, data_size = ys.shape

    d = 10

    func = node.PDEFunc(d=d, width_size=d, depth=2, integrate=False, skew=SKEW) # number of steps taken to solve is very important. Use more advanced method?
    model = node.NeuralODEClassifier(func, in_size=28*28, out_size=10, key=model_key, rtol=1e-2, atol=1e1)
    params = model.get_params()
    model.set_params(params * 1e-3 / jnp.max(jnp.abs(params))) # try to make the ODE less stiff at initialization

    grad_tracker = StatTracker(['adjoint_norm'])

    # Training loop where we train on only length_strategy[i] in the ith iteration
    # This avoids getting stuck in a local minimum

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi, labels):
        y_pred = vmap(model, in_axes=(None, 0, None))(ti, yi, TRACK_STATS)
        return _loss_func(one_hot(labels, 10), y_pred, model, lam=1e0)

    # @eqx.filter_jit
    def make_step(ti, yi, model, opt_state, labels):
        loss, grads = grad_loss(model, ti, yi, labels)
        updates, opt_state = optim.update(grads, opt_state) 

        if DO_BACKWARD: 
            # to_grad = lambda y: _loss_func(one_hot(labels, 10), y, model, lam=1e0)
            # pass loss func, in NODEClf, compute grad(lambda y: _loss_func(labels, self.out_layer(y), self, lam=1e0))
            backward_pass = model.backward(ti, yi, _loss_func, one_hot(labels, 10))
            n_adj = backward_pass.ys.shape[-1] // 2
            adjoint = backward_pass.ys[:, :, :n_adj]
            adjoint_norm = jnp.linalg.norm(adjoint, axis=-1)
            if TRACK_STATS:
                grad_tracker.update({'adjoint_norm': adjoint_norm})

        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    def _celoss(y, y_pred):
        return -jnp.sum(y*jnp.log(y_pred + .001))

    def _loss_func(y, y_pred, model, lam=1e0):
        ce = vmap(_celoss, in_axes=(-1, -1))(y, y_pred) 
        return jnp.mean(ce) # + lam * jnp.mean(jnp.square(model.get_params()))

    for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy):
        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        for step, (yi, labels) in zip( 
            range(steps), training_generator 
        ):
            _ts = jnp.array([0., length])
            start = time.time()
            loss, model, opt_state = make_step(_ts, yi, model, opt_state, labels)
            end = time.time()
            if (step % print_every) == 0 or step == steps - 1:
                print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")
                preds = vmap(model, in_axes=(None, 0, None))(_ts, yi, False)
                preds = jnp.argmax(preds, axis=-1)
                acc = jnp.mean(labels == preds)
                print(f'Train accuracy: {acc}')

                preds = vmap(model, in_axes=(None, 0, None))(_ts, test_images, False)
                preds = jnp.argmax(preds, axis=-1)
                acc = jnp.mean(jnp.argmax(test_labels, axis=-1) == preds)
                print(f'Test accuracy: {acc}') 
    return model, grad_tracker

model, grad_tracker = main()

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

    with open(dst + '/grad_init.pkl', 'wb') as handle:
        to_save = model.get_stats()['grad_init']
        save_jnp(to_save, handle)

    with open( dst + '/adjoint_norm.pkl', 'wb') as handle:
        to_save = grad_tracker.attributes['adjoint_norm']
        save_jnp(to_save, handle)
