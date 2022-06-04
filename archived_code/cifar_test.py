# This file is taken from
# https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html

# TODO: other datasets here: http://proceedings.mlr.press/v80/helfrich18a/helfrich18a.pdf
# https://github.com/ajithcodesit/lstm_copy_task/blob/master/LSTM_copy_task.py 

from turtle import backward
import numpy as np
from torch.utils import data
from torchvision.datasets import CIFAR10
import jax.numpy as jnp
from jax import vmap, grad

from models import NeuralODEClassifier as node
from models.NeuralODE import StatTracker

import equinox as eqx
import optax
import jax.random as jrandom
import jax.nn as jnn
import time
import pickle
import os
import matplotlib.pyplot as plt

from models.nn_with_params import LinearWithParams

TRACK_STATS = True 
DO_BACKWARD = True
SKEW = False 
skew = 'skew' if SKEW else 'none'
dst = f'tests/cifar10_run_{skew}'

n_epochs = 4
batch_size = 64
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

class MyTransform(object):
  def __call__(self, pic):
    mean = np.array([0.4913997551666284, 0.48215855929893703, 0.4465309133731618])
    std = np.array([0.24703225141799082, 0.24348516474564, 0.26158783926049628])
    pic = (pic - mean)/std
    # pic = np.mean(pic, axis=-1)
    # pic = np.reshape(pic, (32, 32, 1))
    # assert pic.shape == (32, 32)
    # return np.ravel(np.array(pic, dtype=jnp.float32))
    # pic = pic[::2, ::2, :]
    return np.transpose(np.array(pic, dtype=jnp.float32), (2, 0, 1))

class Convs(eqx.Module):
    conv: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d 
    lin: eqx.nn.Linear
    pool: eqx.nn.MaxPool2D
    pool2: eqx.nn.MaxPool2D
    in_channels: int
    # pool: eqx.nn.MaxPool2D

    def __init__(self, d, img_size, kernel_size=3, key=None, **kwargs):
        super().__init__(**kwargs)
        key1, key2, key3 = jrandom.split(key, 3)
        in_channels = 3
        self.in_channels = in_channels
        self.conv = eqx.nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=kernel_size, key=key1)
        self.pool = eqx.nn.MaxPool2D(kernel_size=kernel_size, stride=2)
        self.conv2 = eqx.nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=kernel_size, key=key1)
        self.pool2 = eqx.nn.MaxPool2D(kernel_size=kernel_size, stride=2)
        self.lin = eqx.nn.Linear(in_features=self.in_fts(img_size), out_features=d, key=key3)

    def in_fts(self, img_size):
        y = self.conv(jnp.zeros((self.in_channels, img_size, img_size)))
        y = self.pool(y)
        y = jnn.relu(y)
        y = self.conv2(y)
        y = self.pool2(y)
        y = jnp.reshape(y, (-1, ))
        return y.shape[-1]

    def __call__(self, x, *args, **kwargs):
        x = self.conv(x)
        x = self.pool(x)
        x = jnn.relu(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = jnp.reshape(x, (-1, ))
        return self.lin(x)


cifar10_dataset = CIFAR10('/tmp/cifar10/', download=True, transform=MyTransform())
training_generator = NumpyLoader(cifar10_dataset, batch_size=batch_size, num_workers=0)

train_images = np.array(cifar10_dataset.data).reshape(len(cifar10_dataset.data), -1)
train_labels = one_hot(np.array(cifar10_dataset.targets), n_targets)

# idx = 0
# fig, axs = plt.subplots(4, 5)
# for i in range(4):
#     for j in range(5):
#         ax = axs[i, j]
#         ax.imshow(train_images[5*i + j].reshape(32, 32, 3))
#         ax.set_title(np.argmax(train_labels[5*i + j]))
# plt.show()

# Get full test dataset
cifar10_dataset_test = CIFAR10('/tmp/cifar10/', download=True, train=False, transform=MyTransform())
testing_generator = NumpyLoader(cifar10_dataset_test, batch_size=batch_size, num_workers=0)
test_images = np.array(cifar10_dataset_test.data).reshape(len(cifar10_dataset_test.data), -1)
test_labels = one_hot(np.array(cifar10_dataset_test.targets), n_targets)

def main(
    lr=1e-4,
    n_epochs=n_epochs,
    steps_per_epoch=1000,
    seed=0,
    print_every=5,
):  
    key = jrandom.PRNGKey(seed)
    conv_key, model_key = jrandom.split(key, 2)

    # ts, ys = get_data(dataset_size, key=data_key) 
    # _, length_size, data_size = ys.shape

    d = 20 # sort of worked with 20
    width_size = d
    conv = Convs(d, 32, key=conv_key)

    func = node.PDEFunc(d=d, width_size=width_size, depth=2, integrate=False, skew=SKEW) # number of steps taken to solve is very important. Use more advanced method?
    model = node.NeuralODEClassifier(func, in_size=None, out_size=10, key=model_key, rtol=1e-3, atol=1e-6, use_out=True, input_layer=conv)
    params = model.get_params()
    model.set_params(params * 1e-2 / jnp.max(jnp.abs(params))) # try to make the ODE less stiff at initialization

    grad_tracker = StatTracker(['adjoint_norm'])

    first_y = None
    first_lbl = None

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi, labels):
        y_pred = vmap(model, in_axes=(None, 0, None))(ti, yi, TRACK_STATS)
        return _loss_func(one_hot(labels, 10), y_pred, model)

    # @eqx.filter_jit
    def make_step(ti, yi, model, opt_state, labels):
        loss, grads = grad_loss(model, ti, yi, labels)
        updates, opt_state = optim.update(grads, opt_state) 
        if jnp.sum(jnp.abs(grads.get_params())) == 0:
            print('\nNO UPDATE\n')

        if DO_BACKWARD: 
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

    def _loss_func(y, y_pred, model, lam=1e-1):
        ce = vmap(_celoss, in_axes=(-1, -1))(y, y_pred) 
        return jnp.mean(ce)# + lam * jnp.mean(jnp.square(model.get_params()))

    for epoch in range(n_epochs):
        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        steps = steps_per_epoch
        length = 1.
        print(f'\nepoch {epoch+1}/{n_epochs}')
        for step, (yi, labels), (test_img, test_lbl) in zip( 
            range(steps), training_generator, testing_generator
        ):
            # if first_y is None:
            #     first_y = yi 
            #     first_lbl = labels 
            # yi = first_y 
            # labels = first_lbl # overfit on one sample
            # _ts = jnp.array([0., .5*length, length])
            _ts = jnp.linspace(0., length, 100)
            start = time.time()
            loss, model, opt_state = make_step(_ts, yi, model, opt_state, labels)
            end = time.time()
            if (step % print_every) == 0 or step == steps - 1:
                print(f"Step: {step}, Loss: {loss/batch_size}, Computation time: {end - start}")
                # nfe = jnp.mean(model.get_stats()['num_steps'][-1])
                # print(f'mean nfe: {nfe}')
                preds = vmap(model, in_axes=(None, 0, None))(_ts, yi, False)
                preds = jnp.argmax(preds, axis=-1)
                acc = jnp.mean(labels == preds)
                print(f'Train accuracy: {acc}')

                preds = vmap(model, in_axes=(None, 0, None))(_ts, test_img, False)
                preds = jnp.argmax(preds, axis=-1)
                acc = jnp.mean(test_lbl == preds)
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

    # with open(dst + '/grad_init.pkl', 'wb') as handle:
    #     to_save = model.get_stats()['grad_init']
    #     save_jnp(to_save, handle)

    with open( dst + '/adjoint_norm.pkl', 'wb') as handle:
        to_save = grad_tracker.attributes['adjoint_norm']
        save_jnp(to_save, handle)
