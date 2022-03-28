import equinox as eqx
import jax.numpy as jnp 
import jax.random as jrandom
import jax.nn as jnn

class WeightDynamics(eqx.Module):
    d: int 

    def __init__(self, d):
        super().__init__()
        self.d = d 


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
                    activation=jnn.swish,
                    key=key+i, # in order not to initialize identical networks
                ) for i in range(d)] 

    def __call__(self, W):
        d = self.d
        w = jnp.reshape(W, (-1, d*d, ))
        B = [f(w_) for w_, f in zip(w, self.f)]
        B = [jnp.reshape(f, W.shape[-2:]) for f in B]
        B = [f - jnp.transpose(f) for f in B]
        return jnp.array([a*b for a, b in zip(self.a, B)])