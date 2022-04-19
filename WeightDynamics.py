import equinox as eqx
import jax.numpy as jnp 
import jax.random as jrandom
import jax.nn as jnn
from nn_with_params import *

class WeightDynamics(eqx.Module):
    d: int 

    def __init__(self, d):
        super().__init__()
        self.d = d 

    def set_params(self, params: dict):
        pass

class IsoODE(WeightDynamics):
    """
    Implementation of IsoODE from Ode to ODE
    Does not seem to work well
    """
    _Q: jnp.ndarray 
    _N: jnp.ndarray 
    d: int

    def __init__(self, d, key=None, std=1, mean=0, **kwargs):
        super().__init__(d, **kwargs)
        self._Q = mean + std*jrandom.normal(key=key, shape=(d, d))
        self._N = mean + std*jrandom.normal(key=key, shape=(d, d))
        self.d = d

    def __call__(self, W):
        Q = jnp.transpose(self._Q) + self._Q
        N = jnp.transpose(self._N) + self._N
        A = jnp.matmul(jnp.transpose(W), jnp.matmul(Q, W))
        an = jnp.matmul(A, N)
        return an - jnp.transpose(an)



class GatedODE(WeightDynamics):
    """
    Implementation of GatedODE from Ode to ODE
    """
    a: jnp.array # learnable weights for the neural network matrices
    f: list # list of neural networks
    d: int
    width: int
    depth: int
    n_params: int

    def __init__(self, d, width, key=None, depth=2, **kwargs):
        super().__init__(d, **kwargs)
        self.d = d
        self.width = width
        self.depth = depth
        self.a = jrandom.normal(key=key, shape=(d, ))
        self.f = [
            MLPWithParams( 
                in_size=d*d,
                out_size=d*d,
                width_size=width,
                depth=depth,
                activation=jnn.swish,
                key=key+i, # in order not to initialize identical networks
            ) for i in range(d)
        ]
        self.n_params = int(sum([f.n_params for f in self.f])) + d # parameters are f and a

    def __call__(self, W):
        """
        Returns a skew symmetric matrix which is used to update W
        """
        d = self.d
        fs = self.f
        w = jnp.reshape(W, (-1, d*d, ))
        B = [f(w_) for w_, f in zip(w, fs)]
        B = [jnp.reshape(f, W.shape[-2:]) for f in B]
        B = [f - jnp.transpose(f) for f in B]
        return jnp.array([a*b for a, b in zip(self.a, B)])

    def get_params(self, as_dict=False):
        if as_dict:
            params = {}
            for i, f in enumerate(self.f):
                params[i] = f.get_params(as_dict=True)
            return params
        else:
            return jnp.concatenate([l.get_params(as_dict=False) for l in self.f] + [self.a], axis=0)

    def set_params(self, params, as_dict=False):
        if as_dict:
            for f, v in zip(self.f, params.values()):
                f.set_params(v, as_dict=True)
        else:
            assert len(params) == self.n_params
            counter = 0
            for f in self.f:
                p = params[counter:counter+f.n_params]
                f.set_params(p, as_dict=False)
                counter = counter + f.n_params

        

    