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

from torch import isin

# from import_models import *
from models.NeuralODE import *
from models.WeightDynamics import *
from models.Func import *

from jax.config import config

# config.update('jax_disable_jit', True)

TRACK_STATS = True 
WHICH_FUNC = 'ODE2ODEFunc'
DO_BACKWARD = True 
REGULARIZE = False
PLOT = False

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

def dataloader(arrays, batch_size, *, key, cat_dim=2):
    dataset_size, n_timestamps, n_dim = arrays[0].shape
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    # we concatenate some orthogonal matrix to the state, as we use one dynamical system
    # to describe how the weights and state evolves
    cat = jnp.reshape(jnp.concatenate([jnp.eye(cat_dim)]*batch_size*n_timestamps), newshape=(batch_size, n_timestamps, cat_dim**2))
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            if cat_dim > 0:
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

    if WHICH_FUNC == 'ODE2ODEFunc':
        b = GatedODE(data_size, width=4, depth=2, key=key)
        f = DynX()
        func = ODE2ODEFunc(b, f)
        cat_dim = 2
    
    elif WHICH_FUNC == 'PDEFunc':
        func = PDEFunc(d=2, width_size=4, depth=2)
        cat_dim = 0
    else:
        raise NotImplementedError
    
    model = NeuralODE(func=func)
    sym_loss = SymmetricLoss(func)

    grad_tracker = StatTracker(['adjoint_norm'])

    # Training loop where we train on only length_strategy[i] in the ith iteration
    # This avoids getting stuck in a local minimum

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        y_pred = jax.vmap(model, in_axes=(None, 0, None))(ti, yi[:, 0, :], TRACK_STATS)
        if REGULARIZE:
            return _loss_func(yi, y_pred) + jnp.mean(sym_loss(ti, y_pred)) # in this example, only the first two dimensions are the output
        else:
            return _loss_func(yi, y_pred)

    # @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(grads, opt_state) 
        
        if DO_BACKWARD: 
            y_pred = jax.vmap(model, in_axes=(None, 0, None))(ti, yi[:, 0, :], False) 
            dLdT = jax.grad(lambda y: _loss_func(yi, y))(y_pred)[:, -1, :] # end state of the adjoint
            # end_state_loss = jnp.zeros((dLdT.shape[0], model.n_params, ))
            # joint_end_state = jnp.concatenate([dLdT, end_state_loss, y_pred[:, -1, :]], axis=-1)
            joint_end_state = jnp.concatenate([dLdT, y_pred[:, -1, :]], axis=-1)
            backward_pass = jax.vmap(model.backward, in_axes=(None, 0))(ti, joint_end_state)

            n_adj = dLdT.shape[-1]
            adjoint = backward_pass.ys[:, :, :n_adj]

            adjoint_norm = jnp.linalg.norm(adjoint, axis=-1)
            adjoint_errors = jnp.max(adjoint_norm, axis=-1) / jnp.min(adjoint_norm, axis=-1)
            if isinstance(model.func, PDEFunc):
                A = jax.jacrev(lambda z: model.func(1, z, []))(jnp.ones((2, ))) 
                anti_sym_error = jnp.sum(jnp.abs(A + jnp.transpose(A))) # can be large!!


            if TRACK_STATS:
                grad_tracker.update({'adjoint_norm': adjoint_norm})

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
            range(steps), dataloader((_ys,), batch_size, key=loader_key, cat_dim=cat_dim)
        ):
            start = time.time()
            loss, model, opt_state = make_step(_ts, yi, model, opt_state)
            end = time.time()
            if (step % print_every) == 0 or step == steps - 1:
                print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")

    if plot:
        plt.plot(ts, ys[0, :, 0], c="dodgerblue", label="Real")
        plt.plot(ts, ys[0, :, 1], c="dodgerblue")
        # model_y = model(ts, jnp.concatenate([ys[0, 0], jnp.eye(1).reshape((-1, ))]))
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
    plot=PLOT, 
    dataset_size=100,
)

def save_jnp(to_save, handle):
    try:
        pickle.dump([item.val._value for item in to_save], handle)
    except AttributeError:
        try:
            pickle.dump([item._value for item in to_save], handle)
        except AttributeError:
            pickle.dump([item.val.aval.val._value for item in to_save], handle)

if TRACK_STATS:
    with open('outputs/num_steps.pkl', 'wb') as handle:
        to_save = model.get_stats()['num_steps']
        save_jnp(to_save, handle)
        

    with open('outputs/state_norm.pkl', 'wb') as handle:
        to_save = model.get_stats()['state_norm']
        save_jnp(to_save, handle)

    with open('outputs/grad_init.pkl', 'wb') as handle:
        to_save = model.get_stats()['grad_init']
        save_jnp(to_save, handle)

    with open('outputs/adjoint_norm.pkl', 'wb') as handle:
        to_save = grad_tracker.attributes['adjoint_norm']
        save_jnp(to_save, handle)


