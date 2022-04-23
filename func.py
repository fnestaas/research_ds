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
    init_nn: MLPWithParams
    grad_nn: MLPWithParams
    d: int
    n_params: int

    def __init__(self, d: int, width_size: int, depth: int, seed=0, **kwargs) -> None:
        super().__init__(**kwargs)

        self.d = d

        key = jrandom.PRNGKey(seed)
        k1, k2 = jrandom.split(key, 2)
        in_size = d 
        out_size = d 
        self.init_nn = MLPWithParams(in_size, out_size, width_size, depth, key=k1) # predicts initial conditions
        self.grad_nn = MLPWithParams(in_size, out_size*out_size, width_size, depth, key=k2) # predicts gradient of f

        self.n_params = self.init_nn.n_params + self.grad_nn.n_params

    def __call__(self, ts, x, args):
        f0 = self.init_nn(x) # predict initial value of f
        f = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda s, fct, args: self.g_term(s, fct, x, args)),
            diffrax.Tsit5(),
            t0=0.,
            t1=1.,
            dt0=.01, 
            y0=f0,
            stepsize_controller=diffrax.PIDController(),
            saveat=diffrax.SaveAt(t0=False, t1=True),
        )
        out = f.ys.reshape(f0.shape) # + f0 # diffrax adds f0 automatically
        return out


    def g_term(self, t, f, x, args):
        d = self.d
        out = self.grad_nn(t*x) # \nabla f(t*x)
        return jnp.reshape(out, (d, d)) @ x

    def get_params(self, as_dict=False):
        if as_dict:
            raise NotImplementedError
        return jnp.concatenate([self.init_nn.get_params(as_dict=as_dict), self.grad_nn.get_params(as_dict=as_dict)])

    def set_params(self, params, as_dict=False):
        if as_dict:
            raise NotImplementedError 
        divide = self.init_nn.n_params
        self.init_nn.set_params(params[:divide], as_dict=as_dict)
        self.grad_nn.set_params(params[divide:], as_dict=as_dict)