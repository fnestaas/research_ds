from turtle import width
from xmlrpc.client import Boolean, boolean
import jax.numpy as jnp
import jax
import equinox as eqx
from models.nn_with_params import *
from abc import abstractmethod
from equinox.nn.composed import _identity
import matplotlib

class CDEFunc(eqx.Module):
    d: int 
    hidden_size: int

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass


class CDEPDEFunc(CDEFunc):
    init_nn: LinearWithParams
    grad_nn: MLPWithParams
    grad_final: LinearWithParams
    d: int
    hidden_size: int
    n_params: int
    seed: int
    skew: bool # whether to predict using a skew-symmetric matrix
    integrate: bool
    final_activation: Callable
    tau: float # directly influences the adjoint norm when not integrating

    def __init__(self, d: int, hidden_size: int,  width_size: int, depth: int, seed=0, skew=True, final_activation=_identity, integrate=False, tau=1, **kwargs) -> None:
        super().__init__(d, hidden_size, **kwargs)

        self.d = d
        self.seed = seed

        key = jrandom.PRNGKey(seed)
        k1, k2, k3 = jrandom.split(key, 3)

        in_size = hidden_size
        regular_out = d * hidden_size
        self.init_nn = LinearWithParams(1, regular_out, key=k1)

        grad_out = hidden_size * regular_out
        self.grad_nn = MLPWithParams(in_size, width_size, width_size, depth, key=k2, final_activation=lambda x: x) 
        self.grad_final = LinearWithParams(width_size, grad_out, key=k3)

        self.grad_final.set_params(
            self.grad_final.get_params() / jnp.sqrt(2*(hidden_size - int(skew)))
        )
        self.init_nn.set_params(
            self.init_nn.get_params() / jnp.sqrt(2*width_size*d)
        )
        
        self.n_params = self.init_nn.n_params + self.grad_nn.n_params
        self.skew = skew
        self.hidden_size = hidden_size 
        self.integrate = integrate
        self.final_activation = final_activation
        self.tau = tau

    def __call__(self, ts, x, args):
        z = x
        if self.integrate:
            N = 100
            s = jnp.linspace(0, 1, N+1)
            y = jax.vmap(self.integrand, in_axes=(None, 0))(z, s) # A(sx)x
            integral = jnp.trapz(y, x=s, dx=1/N, axis=0) 
        else:
            tau = self.tau
            integral = self.integrand(z, tau) 
        
        return integral + self.pred_init()

    def integrand(self, x, s):
        out = self.pred_mat(x, s) # (hidden_size, d, hidden_size)
        norm = jnp.linalg.norm(x)
        b = 4
        res = jnp.matmul(out, x/jnp.power(1+jnp.power(norm, b), 1/b)) # (hidden_size, d)
        return res

    def pred_mat(self, x, s):
        d = self.d
        out = self.grad_nn(s*x) # \nabla f(s*x)
        out = self.final_activation(self.grad_final(out))
        out = jnp.reshape(out, (self.hidden_size, d, self.hidden_size))
        if self.skew:
            out = out - jnp.transpose(out, (2, 1, 0))
            out = out / jnp.sqrt(2) # should still be okay
        return out # (hidden_size, d, hidden_size)
        
    def pred_init(self):
        return self.init_nn(jnp.array([
            0 # could be time if desired
        ])).reshape((self.hidden_size, self.d))

    def get_params(self):
        return jnp.concatenate([self.init_nn.get_params(), self.grad_nn.get_params()])

    def set_params(self, params):
        n_init = self.init_nn.n_params
        self.init_nn.set_params(params[:n_init])
        self.grad_nn.set_params(params[n_init:])


class CDERegularFunc(CDEFunc):
    nn: MLPWithParams
    d: int
    n_params: int
    seed: int
    hidden_size: int

    def __init__(self, d: int, hidden_size: int, width_size: int, depth: int, seed=0, final_activation=_identity, **kwargs) -> None:
        super().__init__(d, hidden_size, **kwargs)

        self.d = d
        self.seed = seed
        self.hidden_size = hidden_size

        key = jrandom.PRNGKey(seed)
        in_size = hidden_size
        out_size = in_size * d
        self.nn = MLPWithParams(in_size, out_size=out_size, width_size=width_size, depth=depth, key=key, final_activation=final_activation)        

        params = self.nn.get_params()
        k = out_size*(width_size + 1)
        self.nn.set_params(
            jnp.concatenate([
                params[:-k],
                params[-k:] / jnp.sqrt(d)
            ])
        )

        self.n_params = self.nn.n_params

    def __call__(self, ts, x, args):
        return self.nn(x).reshape((self.hidden_size, self.d))

    def get_params(self):
        return self.nn.get_params()

    def set_params(self, params):
        self.nn.set_params(params)
