from logging import warning
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
from equinox.nn.composed import _identity

import pickle

import os

# from import_models import *
from models.NeuralODE import *
from models.WeightDynamics import *
from models.Func import *

from jax.config import config

import argparse

# config.update('jax_disable_jit', True)

# TRACK_STATS = True 
# WHICH_FUNC = 'PDEFunc' # 'PDEFunc' # 'ODE2ODEFunc'
# DO_BACKWARD = True 
# REGULARIZE = False
# PLOT = True
# USE_AUTODIFF = True # uses actual gradients but also allows for checking manual gradient computation


TRACK_STATS = True
WHICH_FUNC = 'RegularFunc' # 'PDEFunc'
DO_BACKWARD = True
REGULARIZE = False
PLOT = True
USE_AUTODIFF = True
SKEW_PDE = True
INTEGRATE = False
FINAL_ACTIVATION = 'identity'
SEED = 0
dst = 'tests/just_a_test'

if not os.path.exists(dst):
    os.makedirs(dst)

def make_np(arr_list):
    try:
        res = [item.val._value for item in arr_list]
    except AttributeError:
        try:
            res = [item._value for item in arr_list]
        except AttributeError:
            res = [item.val.aval.val._value for item in arr_list]
    return res

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
        if FINAL_ACTIVATION == 'identity':
            final_activation = _identity
        elif FINAL_ACTIVATION == 'swish':
            final_activation = jnn.swish
        elif FINAL_ACTIVATION =='sigmoid':
            final_activation = jnn.sigmoid
        elif FINAL_ACTIVATION == 'abs':
            final_activation = lambda x: jnn.relu(x) + jnn.relu(-x)
            import warnings
            warnings.warn('ONLY final activation, not others!')
        else:
            raise NotImplementedError
        width_size = 20
        func = PDEFunc(d=2, width_size=width_size, depth=2, skew=SKEW_PDE, integrate=INTEGRATE, final_activation=final_activation, seed=seed)
        cat_dim = 0
    elif WHICH_FUNC == 'RegularFunc':
        func = RegularFunc(d=2, width_size=2, depth=2, seed=seed)
        cat_dim = 0

    elif WHICH_FUNC == 'PWConstFunc':
        width_size = 4
        func = PWConstFunc(d=2, width_size=width_size, depth=2, seed=seed)
        cat_dim = 0
    else:
        raise NotImplementedError
    
    model = NeuralODE(func=func, keep_grads=not DO_BACKWARD or USE_AUTODIFF)
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
        assert USE_AUTODIFF or DO_BACKWARD, 'we have to use some gradient'
        if USE_AUTODIFF:
            loss, grads = grad_loss(model, ti, yi)
            updates, opt_state = optim.update(grads, opt_state)
        if DO_BACKWARD: 
            # compute gradients "manually"
            y_pred = jax.vmap(model, in_axes=(None, 0, None))(ti, yi[:, 0, :], TRACK_STATS and not USE_AUTODIFF)

            dLdT = jax.grad(lambda y: _loss_func(yi, y))(y_pred)[:, -1, :] # end state of the adjoint
            dldt = grads.get_params()
            if not USE_AUTODIFF:
                end_state_loss = jnp.zeros((dLdT.shape[0], model.n_params, ))
                joint_end_state = jnp.concatenate([dLdT, end_state_loss, y_pred[:, -1, :]], axis=-1)
            else:
                joint_end_state = jnp.concatenate([dLdT, y_pred[:, -1, :]], axis=-1)
            
            backward_pass = jax.vmap(model.backward, in_axes=(None, 0))(ti, joint_end_state)

            n_adj = dLdT.shape[-1]
            adjoint = backward_pass.ys[:, :, :n_adj]

            adjoint_norm = jnp.linalg.norm(adjoint, axis=-1)
            adjoint_errors = jnp.max(adjoint_norm, axis=-1) / jnp.min(adjoint_norm, axis=-1)
            
            if not USE_AUTODIFF:
                computed_grads = -backward_pass.ys[:, :, n_adj:-n_adj]
                cp_grads = jnp.sum(jnp.trapz(computed_grads, axis=1, dx=ts[1]-ts[0]), axis=0)
                scale = 8 # TODO: Why exactly?
                estimated_grad = scale * cp_grads

                # create a gradient to use optim
                if isinstance(model.func, PDEFunc):
                    fc = PDEFunc(model.func.d, model.func.init_nn.width_size, model.func.init_nn.depth)
                else:
                    b = GatedODE(model.func.b.d, model.func.b.width, depth=model.func.b.depth)
                    f = DynX()
                    fc = ODE2ODEFunc(b, f)
                grads = NeuralODE(fc)
                grads.set_params(estimated_grad)
                grads = eqx.filter(grads, eqx.is_array)
                updates, opt_state = optim.update(grads, opt_state)
                
            loss = _loss_func(yi, y_pred)
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
            try:
                start = time.time()
                loss, model, opt_state = make_step(_ts, yi, model, opt_state)
                end = time.time()
                if (step % print_every) == 0 or step == steps - 1:
                    print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")
            except RuntimeError:
                import warnings
                warnings.warn('Max number of function evaluations reached. Aborting.')
                break # give up...

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
    seed=SEED,
)

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


