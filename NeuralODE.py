import jax.numpy as jnp
import jax
import diffrax
import equinox as eqx
from WeightDynamics import * 


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

class StatTracker():
    """
    Keeps track of the statistics specified in attributes
    """
    def __init__(self, attributes):
        self.attributes = {name: [] for name in attributes}

    def update(self, dct):
        for key, val in dct.items():
            self.attributes[key].append(val)
        

class NeuralODE(eqx.Module):
    func: Func
    stats: StatTracker
    n_params: int

    def __init__(self, b, to_track=['num_steps', 'state_norm', 'grad_init'], **kwargs):
        super().__init__(**kwargs)
        f = DynX()
        self.func = Func(b, f, **kwargs)
        self.stats = StatTracker(to_track)
        self.n_params = self.func.n_params

    def solve(self, ts, y0):
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
        return solution

    def __call__(self, ts, y0, update=False): 
        solution = self.solve(ts, y0)
        y_pred = solution.ys
        if update:
            # update the statistics
            self.stats.update(self.compute_stats(solution, ts, y0))
        return y_pred
    
    def compute_stats(self, solution, ts, y0):
        keys = list(self.stats.attributes.keys())
        res = {key: [] for key in keys}
        if 'num_steps' in keys:
            res['num_steps'] = solution.stats['num_steps'].val
        if 'state_norm' in keys:
            y_pred = solution.ys
            res['state_norm'] = jnp.linalg.norm(y_pred).val.aval.val
        if 'grad_init' in keys:
            to_grad = lambda y: self.solve(ts, y)
            res['grad_init'] = jax.jacfwd(to_grad)(y0).ys[-1, :, :].val.aval.val._value
        return res
    
    def get_stats(self, which=None):
        """
        Return recorded statistics of the NeuralODE.
        which is which statistic to return; if none, returns the dict of all statistics
        """
        if which is not None:
            return self.stats.attributes[which]
        else:
            return self.stats.attributes

    def get_params(self, as_dict=False):
        if as_dict:
            return {'func': self.func.get_params(as_dict=True)}
        else: 
            return self.func.get_params(as_dict=False)

    def set_params(self, params, as_dict=False):
        if as_dict:
            self.func.set_params(params['func'], as_dict=True)
        else:
            assert len(params) == self.n_params
            self.func.set_params(params, as_dict=False)
        