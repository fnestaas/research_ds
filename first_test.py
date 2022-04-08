import time
from chex import ArrayTree

import jax 
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom 
import numpy as np
from functools import partial
from jax import jit
import diffrax
import equinox as eqx
import matplotlib.pyplot as plt
import optax

import pickle

from WeightDynamics import * 
from NeuralODE import *
import loss_change
import other_loss_change

def _get_data(ts, *, key):
    y0 = jrandom.uniform(key, (2,), minval=-0.6, maxval=1)

    def f(t, y, args):
        x = y / (1 + y)
        return jnp.stack([x[1], -x[0]], axis=-1)

    solver = diffrax.Tsit5()
    dt0 = 0.1
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(f), solver, ts[0], ts[-1], dt0, y0, saveat=saveat
    )
    ys = sol.ys
    return ys


def get_data(dataset_size, *, key):
    ts = jnp.linspace(0, 10, 100)
    key = jrandom.split(key, dataset_size)
    ys = jax.vmap(lambda key: _get_data(ts, key=key))(key)
    return ts, ys

def dataloader(arrays, batch_size, *, key):
    dataset_size, n_timestamps, n_dim = arrays[0].shape
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    # we concatenate some orthogonal matrix to the state, as we use one dynamical system
    # to describe how the weights and state evolves
    cat = jnp.reshape(jnp.concatenate([jnp.eye(n_dim)]*batch_size*n_timestamps), newshape=(batch_size, n_timestamps, n_dim**2))
    # a = 1/jnp.sqrt(2)
    # b = -a
    # cat = jnp.reshape(jnp.concatenate([jnp.array([[a, b], [-b, a]])]*batch_size*n_timestamps), newshape=(batch_size, n_timestamps, n_dim**2))
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(jnp.concatenate([array[batch_perm], cat], axis=-1) for array in arrays)
            start = end
            end = start + batch_size

def main(
    dataset_size=256,
    batch_size=32,
    lr_strategy=(3e-3, 3e-3),
    steps_strategy=(500, 500),
    length_strategy=(0.1, 1),
    seed=5678,
    plot=True,
    print_every=100,
):
    key = jrandom.PRNGKey(seed)
    data_key, model_key, loader_key = jrandom.split(key, 3)

    ts, ys = get_data(dataset_size, key=data_key)
    _, length_size, data_size = ys.shape

    b = GatedODE(data_size, width=4, depth=2, key=key)
    model = NeuralODE(b=b)

    norm_tracker = StatTracker(['loss_change'])

    # Training loop like normal.
    #
    # Only thing to notice is that up until step 500 we train on only the first 10% of
    # each time series. This is a standard trick to avoid getting caught in a local
    # minimum.

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        y_pred = jax.vmap(model, in_axes=(None, 0, None))(ti, yi[:, 0, :], True) 
        return jnp.mean((yi[:, :, :2] - y_pred[:, :, :2]) ** 2) # in this example, only the first two dimensions are the output

    # @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(grads, opt_state)
        # dLdt = loss_change.loss_change(model, grads, jnp.linspace(0, ti[-1], 10), yi)
        dLdt = other_loss_change.loss_change_other(yi, ti, lambda x, xx: jnp.mean((x[:, :, :2] - xx[:, :, :2])**2), grads, model)
        # norm_tracker.update({'loss_change': jnp.linalg.norm(dLdt, axis=-1)._value})
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy):
        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        _ts = ts[: int(length_size * length)]
        _ys = ys[:, : int(length_size * length)] # length is the fraction of timestamps on which we train
        for step, (yi,) in zip( 
            range(steps), dataloader((_ys,), batch_size, key=loader_key)
        ):
            start = time.time()
            loss, model, opt_state = make_step(_ts, yi, model, opt_state)
            end = time.time()
            if (step % print_every) == 0 or step == steps - 1:
                print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")

    if plot:
        plt.plot(ts, ys[0, :, 0], c="dodgerblue", label="Real")
        plt.plot(ts, ys[0, :, 1], c="dodgerblue")
        model_y = model(ts, jnp.concatenate([ys[0, 0], jnp.array([1, 0, 0, 1])]))
        plt.plot(ts, model_y[:, 0], c="crimson", label="Model")
        plt.plot(ts, model_y[:, 1], c="crimson")
        plt.legend()
        plt.tight_layout()
        plt.savefig("neural_ode2ode.png")
        plt.show()

    return ts, ys, model, norm_tracker


ts, ys, model, norm_tracker = main(
    steps_strategy=(200, 200),
    print_every=100,
    batch_size=4,
    length_strategy=(.1, 1),
    lr_strategy=(3e-3, 1e-3),
    plot=True, 
    dataset_size=100,
)

with open('outputs/num_steps.pkl', 'wb') as handle:
    pickle.dump(model.get_stats()['num_steps'], handle)

with open('outputs/state_norm.pkl', 'wb') as handle:
    pickle.dump(model.get_stats()['state_norm'], handle)

with open('outputs/grad_init.pkl', 'wb') as handle:
    pickle.dump(model.get_stats()['grad_init'], handle)

with open('outputs/stats.pkl', 'wb') as handle:
    pickle.dump(model.get_stats(), handle)

with open('outputs/grad_info.pkl', 'wb') as handle:
    pickle.dump(norm_tracker.attributes['loss_change'], handle)


