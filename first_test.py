import time

import jax 
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom 
from functools import partial
from jax import jit

import diffrax
import equinox as eqx

import matplotlib.pyplot as plt

import optax

import copy

class IsoODE(eqx.Module):
    _Q: jnp.ndarray 
    _N: jnp.ndarray 
    d: int

    def __init__(self, d, key=None, std=1, mean=0, **kwargs):
        super().__init__(**kwargs)
        self._Q = mean + std*jrandom.normal(key=key, shape=(d, d))
        self._N = mean + std*jrandom.normal(key=key, shape=(d, d))
        self.d = d

    # @partial(jit, static_argnums=1)
    def __call__(self, W):
        Q = jnp.transpose(self._Q) + self._Q
        N = jnp.transpose(self._N) + self._N
        A = jnp.matmul(jnp.transpose(W), jnp.matmul(Q, W))
        an = jnp.matmul(A, N)
        return an - jnp.transpose(an)

class GatedODE(eqx.Module):
    a: jnp.array # learnable weights for the neural network matrices
    f: list # list of neural networks
    d: int

    def __init__(self, d, width, key=None, depth=2, **kwargs):
        super().__init__(**kwargs)
        self.d = d
        self.a = jrandom.normal(key=key, shape=(d, ))
        self.f = [eqx.nn.MLP(
                    in_size=d*d,
                    out_size=d*d,
                    width_size=width,
                    depth=depth,
                    activation=jnn.softplus,
                    key=key+i, # in order not to initialize identical networks
                ) for i in range(d)] 

    def __call__(self, W):
        d = self.d
        w = jnp.reshape(W, (-1, d*d, ))
        f_test = self.f[0](w[0, :])
        B = [f(w_) for w_, f in zip(w, self.f)]
        B = [jnp.reshape(f, W.shape[-2:]) for f in B]
        B = [f - jnp.transpose(f) for f in B]
        return jnp.array([a*b for a, b in zip(self.a, B)])



class DynX(eqx.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self, x):
        return jnp.abs(x)

class Func(eqx.Module):
    # b: IsoODE
    b: GatedODE
    f: DynX
    d: int

    def __init__(self, b, f, **kwargs):
        super().__init__(**kwargs)
        # dynamics by which W_t should evolve
        self.b = b
        self.d = b.d # dimension of x_t
        # dynamics by which x_t should evolve
        self.f = f

    def __call__(self, t, y, args):
        d = self.d
        x = y[:d] 
        W = jnp.reshape(y[d:], newshape=(d, d))
        f = self.f(jnp.matmul(W, x))
        bw = jnp.matmul(self.b(W), W)
        
        return jnp.concatenate([f, jnp.reshape(bw, newshape=(d*d))], axis=0)

class NeuralODE(eqx.Module):
    func: Func

    def __init__(self, data_size, key, **kwargs):
        super().__init__(**kwargs)
        # b = IsoODE(data_size, key=key, **kwargs)
        b = GatedODE(data_size, width=data_size**2, depth=1, key=key, **kwargs)
        f = DynX()
        self.func = Func(b, f, **kwargs)

    def __call__(self, ts, y0): 
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys

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

    model = NeuralODE(data_size, key=model_key)

    # Training loop like normal.
    #
    # Only thing to notice is that up until step 500 we train on only the first 10% of
    # each time series. This is a standard trick to avoid getting caught in a local
    # minimum.

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        # test_call = model(ti, yi[0, 0])
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0, :]) 
        return jnp.mean((yi[:, :, :2] - y_pred[:, :, :2]) ** 2) 

    # @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(grads, opt_state)
        old_model = copy.deepcopy(model)
        model = eqx.apply_updates(model, updates)
        # print(repr(old_model.func.b._Q - model.func.b._Q[0, 0])) # this is traced if we use @eqx.filter_jit
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
        plt.savefig("neural_ode.png")
        plt.show()

    return ts, ys, model


ts, ys, model = main(
    steps_strategy=(300, 300),
    print_every=100,
    length_strategy=(.3, 1),
    lr_strategy=(3e-3, 1e-3),
)