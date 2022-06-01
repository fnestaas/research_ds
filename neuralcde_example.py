import math
import time

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jsp
import matplotlib
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax
from models.NeuralCDE import NeuralCDE, CDEPDEFunc, CDERegularFunc
# from models.Func import PDEFunc, RegularFunc

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

def get_data(dataset_size, add_noise, cat_dim=None, *, key):

    theta_key, noise_key = jrandom.split(key, 2)
    length = 100
    theta = jrandom.uniform(theta_key, (dataset_size,), minval=0, maxval=2 * math.pi)
    y0 = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    ts = jnp.broadcast_to(jnp.linspace(0, 4 * math.pi, length), (dataset_size, length))
    matrix = jnp.array([[-0.3, 2], [-2, -0.3]])
    ys = jax.vmap(
        lambda y0i, ti: jax.vmap(lambda tij: jsp.linalg.expm(tij * matrix) @ y0i)(ti)
    )(y0, ts)
    if cat_dim is not None:
        cat = jnp.concatenate([jnp.ones((ys.shape[1], ))]*cat_dim*dataset_size).reshape((dataset_size, length, cat_dim))
        ys = jnp.concatenate([ys, cat], axis=-1)
    ys = jnp.concatenate([ts[:, :, None], ys], axis=-1)  # time is a channel
    ys = ys.at[: dataset_size // 2, :, 1].multiply(-1)
    if add_noise:
        ys = ys + jrandom.normal(noise_key, ys.shape) * 0.1
    coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, ys)
    labels = jnp.zeros((dataset_size,))
    labels = labels.at[: dataset_size // 2].set(1.0)
    _, _, data_size = ys.shape
    return ts, coeffs, labels, data_size

def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size

def main(
    dataset_size=256,
    add_noise=False,
    batch_size=32,
    lr=1e-3,
    steps=1000,
    seed=56791,
    cat_dim=None
):
    key = jrandom.PRNGKey(seed)
    train_data_key, test_data_key, model_key, loader_key = jrandom.split(key, 4)

    ts, coeffs, labels, data_size = get_data(
        dataset_size, add_noise, cat_dim, key=train_data_key
    )
    cat_dim = 0 if cat_dim is None else cat_dim
    d = 3 + cat_dim # effectively what the model sees, dimension(y) + 1 because of time
    width_size = 64
    depth = 4
    hidden_size = 8
    final_activation = lambda x: x 
    # final_activation = jnn.tanh
    integrate = False
    skew = True
    which_func = 'PDEFunc'
    # which_func = 'RegularFunc'

    if which_func != 'PDEFunc':
        func = CDERegularFunc(d=d, hidden_size=hidden_size, width_size=width_size, depth=depth, seed=seed, final_activation=final_activation)
    else:
        func = CDEPDEFunc(d=d, hidden_size=hidden_size, width_size=width_size, depth=depth, seed=seed, skew=skew, final_activation=final_activation, integrate=integrate)
    
    model = NeuralCDE(d, width_size=width_size, depth=depth, hidden_size=hidden_size, key=model_key, func=func)
    # model.set_params(model.get_params()*1e-6)

    # Training loop like normal.

    def loss_func(label_i, pred, eps=1e-6):
        bxe = label_i * jnp.log(pred + eps) + (1 - label_i) * jnp.log(1 - pred + eps)
        bxe = -jnp.mean(bxe)
        return bxe

    # @eqx.filter_jit
    def loss(model, ti, label_i, coeff_i):
        pred = jax.vmap(model)(ti, coeff_i)
        # Binary cross-entropy
        bxe = loss_func(label_i, pred)
        acc = jnp.mean((pred > 0.5) == (label_i == 1))
        return bxe, acc

    grad_loss = eqx.filter_value_and_grad(loss, has_aux=True)

    def score_adjoint(adjoint, lower = .15):
        upper = 1-lower
        return jnp.quantile(adjoint, upper, axis=-1)/jnp.quantile(adjoint, lower, axis=-1)

    # @eqx.filter_jit
    def make_step(model, data_i, opt_state):
        ti, label_i, *coeff_i = data_i
        (bxe, acc), grads = grad_loss(model, ti, label_i, coeff_i)
        updates, opt_state = optim.update(grads, opt_state)

        backward_pass = jax.vmap(
            lambda t, coeff, label: model.backward(t, coeff, loss_func, label), 
        )(ti, coeff_i, label_i)
        adjoint_norm = jnp.linalg.norm(backward_pass.ys[:, :, :model.func.hidden_size], axis=-1)
        # pred, sol = jax.vmap(model)(ti, coeff_i, evolving_out=True)
        # error = sol.ys - backward_pass.ys[:, ::-1, model.func.hidden_size:]
        print('mean adjoint norm', jnp.mean(adjoint_norm))
        print('adjoint std', jnp.median(jnp.std(adjoint_norm, axis=-1)))
        print('adjoint score', jnp.median(score_adjoint(adjoint_norm)))
        # idx = jnp.argmax(jnp.std(adjoint_norm, axis=-1))
        # plt.plot(adjoint_norm[idx, :])
        # plt.show()
        model = eqx.apply_updates(model, updates)
        return bxe, acc, model, opt_state

    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    for step, data_i in zip(
        range(steps), dataloader((ts, labels) + coeffs, batch_size, key=loader_key)
    ):
        start = time.time()
        bxe, acc, model, opt_state = make_step(model, data_i, opt_state)
        end = time.time()
        print(
            f"Step: {step}, Loss: {bxe}, Accuracy: {acc}, Computation time: "
            f"{end - start}"
        )
        # print(
        #     'num_steps', jnp.max(model.stats.attributes['num_steps'][-1].val)
        # ) 

        # print(
        #     'state norm', jnp.max(model.stats.attributes['state_norm'][-1].val.primal)
        # )
        # print(
        #     'state norm std', jnp.std(model.stats.attributes['state_norm'][-1].val.primal) # not really relevant, we care about the adjoint
        # )
    ts, coeffs, labels, _ = get_data(dataset_size, add_noise, key=test_data_key)
    bxe, acc = loss(model, ts, labels, coeffs)
    print(f"Test loss: {bxe}, Test Accuracy: {acc}")

    # Plot results
    # sample_ts = ts[-1]
    # sample_coeffs = tuple(c[-1] for c in coeffs)
    # pred = model(sample_ts, sample_coeffs, evolving_out=True)
    # interp = diffrax.CubicInterpolation(sample_ts, sample_coeffs)
    # values = jax.vmap(interp.evaluate)(sample_ts)
    # fig = plt.figure(figsize=(16, 8))
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    # ax1.plot(sample_ts, values[:, 1], c="dodgerblue")
    # ax1.plot(sample_ts, values[:, 2], c="dodgerblue", label="Data")
    # ax1.plot(sample_ts, pred, c="crimson", label="Classification")
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    # ax1.set_xlabel("t")
    # ax1.legend()
    # ax2.plot(values[:, 1], values[:, 2], c="dodgerblue", label="Data")
    # ax2.plot(values[:, 1], values[:, 2], pred, c="crimson", label="Classification")
    # ax2.set_xticks([])
    # ax2.set_yticks([])
    # ax2.set_zticks([])
    # ax2.set_xlabel("x")
    # ax2.set_ylabel("y")
    # ax2.set_zlabel("Classification")
    # plt.tight_layout()
    # plt.savefig("neural_cde.png")
    # plt.show()

main()
