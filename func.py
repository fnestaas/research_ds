import equinox as eqx
from WeightDynamics import * 
from nn_with_params import *
import diffrax


class DynX(eqx.Module):
    n_params: int

    """
    Dynamics by which the state evolves
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_params = 0

    def __call__(self, x):
        return x

    def get_params(self, as_dict=False):
        return None
    
    def set_params(self, params, as_dict=False):
        pass



class Func(eqx.Module):
    """
    Complete dynamics of the system; keeps track of how the weights and state evolve.
    """
    b: WeightDynamics
    f: DynX
    d: int
    n_params: int

    def __init__(self, b, f, **kwargs):
        super().__init__(**kwargs)
        # dynamics by which W_t should evolve
        self.b = b
        self.d = b.d # dimension of x_t
        # dynamics by which x_t should evolve
        self.f = f
        self.n_params = b.n_params + f.n_params

    def __call__(self, t, y, args):
        d = self.d
        x = y[:d] 
        W = jnp.reshape(y[d:], newshape=(d, d))
        f = self.f(jnp.matmul(W, x))
        bw = jnp.matmul(W, self.b(W))
        
        return jnp.concatenate([f, jnp.reshape(bw, newshape=(d*d))], axis=0)

    def get_params(self, as_dict=False):
        if as_dict:
            params = {}
            params['b'] = self.b.get_params(as_dict=True)
            params['f'] = self.f.get_params(as_dict=True)
            return params
        else:
            return self.b.get_params(as_dict=False) # ignore f in this case

    def set_params(self, params, as_dict=False):
        if as_dict:
            self.b.set_params(params['b'])
            if 'f' in params.keys():
                self.f.set_params(params['f'], as_dict=True)
        else:
            assert len(params) == self.n_params
            self.b.set_params(params[:self.b.n_params], as_dict=False)
            if self.b.n_params < len(params):
                print('Setting params of f')
                self.f.set_params(params[self.b.n_params:], as_dict=False)

class PDEFunc(eqx.Module):
    # init_nn: MLPWithParams
    grad_nn: MLPWithParams
    d: int
    n_params: int
    L: float

    def __init__(self, d: int, L: float, width_size: int, depth: int, seed=0, **kwargs) -> None:
        super().__init__(**kwargs)

        self.d = d
        self.L = L

        key = jrandom.PRNGKey(seed)
        k1, k2 = jrandom.split(key, 2)
        in_size = d + 1
        out_size = d + 1
        # self.init_nn = MLPWithParams(in_size+1, out_size+1, width_size, depth, key=k1) # predicts initial conditions
        grad_out = int((d + 1) * d / 2) # number of parameters for skew-symmetric matrix of shape (d, d)
        self.grad_nn = MLPWithParams(in_size, grad_out, width_size, depth, key=k2) # predicts gradient of f

        # self.n_params = self.init_nn.n_params + self.grad_nn.n_params
        self.n_params = self.grad_nn.n_params

    def __call__(self, ts, x, args):
        # integrate
        z = jnp.concatenate([x, jnp.array([self.L])])
        y = jax.vmap(self.integrand, in_axes=(None, 0))(z, jnp.linspace(0, 1, 101)) # A(sx)x
        integral = jnp.trapz(y, dx=.01, axis=0)

        assert integral.shape == (self.d+1, ), f'shape of integral is {integral.shape}'

        return integral[:self.d] # + self.integrand(z, 0)[:self.d]

    def integrand(self, x, s):
        d = self.d + 1
        out = self.grad_nn(s*x) # \nabla f(s*x)
        out = jnp.concatenate([out, jnp.zeros(d*d - out.shape[0])]) # make conformable to (d, d)-matrix
        out = jnp.reshape(out, (d, d))
        out = out - jnp.transpose(out)
        return out @ x

    def get_params(self, as_dict=False):
        if as_dict:
            raise NotImplementedError
        # return jnp.concatenate([self.init_nn.get_params(as_dict=as_dict), self.grad_nn.get_params(as_dict=as_dict)])
        return self.grad_nn.get_params(as_dict=as_dict)

    def set_params(self, params, as_dict=False):
        if as_dict:
            raise NotImplementedError 
        divide = self.init_nn.n_params
        # self.init_nn.set_params(params[:divide], as_dict=as_dict)
        self.grad_nn.set_params(params[divide:], as_dict=as_dict)