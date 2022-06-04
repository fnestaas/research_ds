from xmlrpc.client import Boolean, boolean
import jax.numpy as jnp
import jax
import diffrax
import equinox as eqx
from numpy import diff
from models.WeightDynamics import * 
from models.Func import *

class StatTracker():
    """
    Keeps track of the statistics specified in attributes
    """
    def __init__(self, attributes):
        self.attributes = {name: [] for name in attributes}

    def update(self, dct):
        for key, val in dct.items():
            self.attributes[key].append(val)
        
class BackwardPasser(eqx.Module):
    """
    Class for computing the adjoint by integrating backwards
    """
    func: Func
    d: int 


    def __init__(self, func, **kwargs) -> None:
        super().__init__(**kwargs)
        self.func = func
        self.d = func.d


    def ode2ode_func_with_params(self, params):
        """
        Generate a Func with parameters params;
        useful e.g. when computing the derivative of such a Func wrt its parameters
        """
        key = jrandom.PRNGKey(0)
        b = GatedODE(self.d, width=self.func.b.width, depth=self.func.b.depth, key=key)
        f = DynX()
        func = ODE2ODEFunc(b, f)
        func.set_params(params)
        return func

    def full_term_ode2ode_func(self, t, joint_state, args):
        """
        Callable term to perform a backward pass to compute the adjoint, parameter gradients
        as a function of time, and the solution itself (needed to compute the others)
        
        returns [-a d func / d z, a d func / d theta, func] 
        """
        n = self.d * (self.d + 1)
        adjoint = joint_state[:n]
        x = joint_state[-n:]
        grad_comp = (2*n < joint_state.shape[0])

        dfdz = jax.jacfwd(lambda z: self.func(t, z, args))

        t1 = - adjoint @ dfdz(x)
        t3 = self.func(t, x, args)

        if grad_comp:
            dfdth = jax.jacfwd(lambda th: self.ode2ode_func_with_params(th)(t, x, args))
            t2 = adjoint @ dfdth(self.func.get_params())  
            return jnp.concatenate([t1, t2, t3]) # don't negate; diffrax does that
        else:
            return jnp.concatenate([t1, t2])

    def func_with_params(self, params):
        func = PDEFunc(self.func.d, self.func.grad_nn.width_size, self.func.grad_nn.depth)
        func.set_params(params)
        return func

    def full_term_func(self, t, joint_state, args):
        n = self.func.d
        adjoint = joint_state[:n]
        x = joint_state[-n:]
        grad_comp = (2*n < joint_state.shape[0])

        dfdz = jax.jacfwd(lambda z: self.func(t, z, args))
        
        t1 = - adjoint @ dfdz(x)
        t3 = self.func(t, x, args)
        if grad_comp: # compute the gradients "manually"
            dfdth = jax.jacfwd(lambda th: self.func_with_params(th)(t, x, args))
            t2 = adjoint @ dfdth(self.func.get_params())
            return jnp.concatenate([t1, t2, t3])
        else:
            return jnp.concatenate([t1, t3])

    def backward(self, ts, joint_end_state):
        """
        Perform a backward pass through the NeuralODE
        joint_end_state is the final state, it contains the values of the
        adjoint, loss gradient wrt the parameters and state at the end time
        """
        if isinstance(self.func, ODE2ODEFunc):
            term = self.full_term_ode2ode_func
        else:
            term = self.full_term_func

        saveat = diffrax.SaveAt(ts=ts[::-1]) # diffrax doesn't work otherwise
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(term),
            diffrax.Tsit5(),
            t0=ts[-1],
            t1=ts[0],
            dt0=-(ts[1]-ts[0]),
            y0=joint_end_state,
            stepsize_controller=diffrax.PIDController(),
            saveat=saveat,
            max_steps=2**14,
        )
        return solution

class NeuralODE(eqx.Module):
    """
    Neural ODE that can track statistics and perform forward and backward passes to 
    copmute solutions to an initial state, the adjoint, and gradients of the parameters
    """
    func: Func
    stats: StatTracker
    n_params: int
    d: int
    backwardpasser: BackwardPasser
    keep_grads: Boolean
    rtol: float 
    atol: float

    def __init__(self, func, to_track=['num_steps', 'state_norm', 'grad_init'], keep_grads=True, rtol=1e-3, atol=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.func = func
        self.stats = StatTracker(to_track)
        self.n_params = self.func.n_params
        self.d = self.func.d
        self.backwardpasser = BackwardPasser(self.func, **kwargs)
        self.keep_grads = keep_grads
        self.rtol = rtol
        self.atol = atol
        
    def solve(self, ts, y0):
        """
        Compute solution at times ts with initial state y0
        """
        if len(ts)>1:
            dt0 = ts[1] - ts[0] if len(ts) > 2 else (ts[-1] - ts[0]) / 100 # make sure initial stepsize is not too small
        else:
            dt0 = ts[-1] / 100
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=dt0, 
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution

    def __call__(self, ts, y0, update=False): 
        """
        Compute the solution according to this NeuralODE at times ts with
        initial state y0. The parameter update specifies whether to save the
        solver statistics in this pass.
        """
        solution = self.solve(ts, y0)
        y_pred = solution.ys
        if update:
            # update the statistics
            self.stats.update(self.compute_stats(solution, ts, y0))
        return y_pred
    
    def compute_stats(self, solution, ts, y0):
        """
        Updates the statistics contained in self.stats
        """
        keys = list(self.stats.attributes.keys())
        res = {key: [] for key in keys}
        if 'num_steps' in keys:
            res['num_steps'] = solution.stats['num_steps']
        if 'state_norm' in keys:
            y_pred = solution.ys
            res['state_norm'] = jnp.linalg.norm(y_pred, axis=-1)
        if 'grad_init' in keys: # gradient with respect to initial state
            to_grad = lambda y: self.solve(ts, y)
            res['grad_init'] = jax.jacfwd(to_grad)(y0).ys[-1, :]
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

    def backward(self, ts, joint_end_state):
        return self.backwardpasser.backward(ts, joint_end_state)

    def get_params(self):
        return self.func.get_params()

    def set_params(self, params):
        if self.n_params is not None:
            assert len(params) == self.n_params
        self.func.set_params(params)
