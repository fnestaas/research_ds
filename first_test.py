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
from func import *

TRACK_STATS = False 
WHICH_FUNC = 'PDEFunc'
DO_BACKWARD = False

def _get_data(ts, *, key):
    y0 = jrandom.uniform(key, (2,), minval=-0.6, maxval=1)

    def f(t, y, args):
        x = y / (1 + y)
        return jnp.stack([x[1], -x[0]], axis=-1)

    solver = diffrax.Tsit5()
    dt0 = 0.1
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve( # TODO: is this expensive?
        diffrax.ODETerm(f), solver, ts[0], ts[-1], dt0, y0, saveat=saveat
    )
    ys = sol.ys
    return ys

def get_data(dataset_size, *, key):
    ts = jnp.linspace(0, 10, 100)
    key = jrandom.split(key, dataset_size)
    ys = jax.vmap(lambda key: _get_data(ts, key=key))(key)
    return ts, ys

def dataloader(arrays, batch_size, *, key, concat=True):
    dataset_size, n_timestamps, n_dim = arrays[0].shape
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    # we concatenate some orthogonal matrix to the state, as we use one dynamical system
    # to describe how the weights and state evolves
    if concat:
        cat = jnp.reshape(jnp.concatenate([jnp.eye(n_dim)]*batch_size*n_timestamps), newshape=(batch_size, n_timestamps, n_dim**2))
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            if concat:
                yield tuple(jnp.concatenate([array[batch_perm], cat], axis=-1) for array in arrays)
            else:
                yield tuple(array[batch_perm] for array in arrays)
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

    if WHICH_FUNC == 'Func':
        b = GatedODE(data_size, width=4, depth=2, key=key)
        f = DynX()
        func = Func(b, f)
    
    elif WHICH_FUNC == 'PDEFunc':
        func = PDEFunc(d=2, width_size=4, depth=2)
    
    model = NeuralODE(func=func)

    grad_tracker = StatTracker(['loss_change'])

    # Training loop where we train on only length_strategy[i] in the ith iteration
    # This avoids getting stuck in a local minimum

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        y_pred = jax.vmap(model, in_axes=(None, 0, None))(ti, yi[:, 0, :], TRACK_STATS)
        return _loss_func(yi, y_pred) # in this example, only the first two dimensions are the output

    # @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(grads, opt_state)

        if DO_BACKWARD: # TODO: makes more sense to not compute grads in grad_loss in this case; skip that computation in that case
            y_pred = jax.vmap(model, in_axes=(None, 0, None))(ti, yi[:, 0, :], False) 
            dLdT = jax.grad(lambda y: _loss_func(yi, y))(y_pred)[0, -1, :] # end state of the adjoint
            end_state_loss = jnp.zeros((model.n_params, ))
            joint_end_state = jnp.concatenate([dLdT, end_state_loss, y_pred[0, 0, :]], axis=-1)
            backward_pass = model.backward(ti, joint_end_state)
        
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    def _loss_func(y, y_pred):
        return jnp.mean((y[:, :, :2] - y_pred[:, :, :2]) ** 2)

    for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy):
        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        _ts = ts[: int(length_size * length)]
        _ys = ys[:, : int(length_size * length)] # length is the fraction of timestamps on which we train
        for step, (yi,) in zip( 
            range(steps), dataloader((_ys,), batch_size, key=loader_key, concat=False)
        ):
            start = time.time()
            loss, model, opt_state = make_step(_ts, yi, model, opt_state)
            end = time.time()
            if (step % print_every) == 0 or step == steps - 1:
                print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")

    if plot:
        plt.plot(ts, ys[0, :, 0], c="dodgerblue", label="Real")
        plt.plot(ts, ys[0, :, 1], c="dodgerblue")
        # model_y = model(ts, jnp.concatenate([ys[0, 0], jnp.array([1, 0, 0, 1])]))
        model_y = model(ts, ys[0, 0])
        plt.plot(ts, model_y[:, 0], c="crimson", label="Model")
        plt.plot(ts, model_y[:, 1], c="crimson")
        plt.legend()
        plt.tight_layout()
        plt.savefig("neural_ode2ode.png")
        plt.show()

    return ts, ys, model, grad_tracker


ts, ys, model, grad_tracker = main(
    steps_strategy=(200, 200),
    print_every=100,
    batch_size=50,
    length_strategy=(.1, 1),
    lr_strategy=(3e-3, 1e-3),
    plot=True, 
    dataset_size=100,
)

if TRACK_STATS:
    with open('outputs/num_steps.pkl', 'wb') as handle:
        pickle.dump(model.get_stats()['num_steps'], handle)

    with open('outputs/state_norm.pkl', 'wb') as handle:
        pickle.dump(model.get_stats()['state_norm'], handle)

    with open('outputs/grad_init.pkl', 'wb') as handle:
        pickle.dump(model.get_stats()['grad_init'], handle)

    with open('outputs/stats.pkl', 'wb') as handle:
        pickle.dump(model.get_stats(), handle)

    with open('outputs/grad_info.pkl', 'wb') as handle:
        pickle.dump(grad_tracker.attributes['loss_change'], handle)


