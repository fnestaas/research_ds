from xmlrpc.client import boolean
import jax.numpy as jnp
import jax
import diffrax
import equinox as eqx
from WeightDynamics import * 
from func import *

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
    """
    Neural ODE that can track statistics and perform forward and backward passes to 
    copmute solutions to an initial state, the adjoint, and gradients of the parameters
    """
    func: Func
    stats: StatTracker
    n_params: int
    d: int

    def __init__(self, func, to_track=['num_steps', 'state_norm', 'grad_init'], **kwargs):
        super().__init__(**kwargs)
        # f = DynX() # function that specifies \dot{x} = f(Wx)
        # self.func = Func(b, f, **kwargs) # function that specifies the complete system dynamics
        self.func = func
        self.stats = StatTracker(to_track)
        self.n_params = self.func.n_params
        self.d = self.func.d

    def solve(self, ts, y0):
        """
        Compute solution at times ts with initial state y0
        """
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

    def func_with_params(self, params):
        """
        Generate a Func with parameters params;
        useful e.g. when computing the derivative of such a Func wrt its parameters
        """
        key = jrandom.PRNGKey(0)
        b = GatedODE(self.d, width=self.func.b.width, depth=self.func.b.depth, key=key)
        f = DynX()
        func = Func(b, f)
        func.set_params(params, as_dict=False)
        return func

    def full_term_func(self, t, joint_state, args):
        """
        Callable term to perform a backward pass to compute the adjoint, parameter gradients
        as a function of time, and the solution itself (needed to compute the others)
        
        returns [a d func / d z, -a d func / d theta, -func] 
        """
        n = self.d * (self.d + 1)
        adjoint = joint_state[:n]
        x = joint_state[-n:]

        dfdz = jax.jacrev(lambda z: self.func(t, z, args))
        dfdth = jax.jacrev(lambda th: self.func_with_params(th)(t, x, args))

        t1 = adjoint @ dfdz(x)
        t2 = - adjoint @ dfdth(self.func.get_params(as_dict=False))
        t3 = - self.func(t, x, args)

        return jnp.concatenate([t1, t2, t3])

    def pdefunc_with_params(self, params):
        func = PDEFunc(self.func.d, self.func.L, self.func.grad_nn.width_size, self.func.grad_nn.depth)
        func.set_params(params, as_dict=False)
        return func

    def full_term_pdefunc(self, t, joint_state, args):
        n = self.func.d
        adjoint = joint_state[:n]
        x = joint_state[-n:]

        dfdz = jax.jacrev(lambda z: self.func(t, z, args))
        dfdth = jax.jacrev(lambda th: self.pdefunc_with_params(th)(t, x, args))

        t1 = adjoint @ dfdz(x)
        t2 = - adjoint @ dfdth(self.func.get_params(as_dict=False))
        t3 = - self.func(t, x, args)

        return jnp.concatenate([t1, t2, t3])

    def backward(self, ts, joint_end_state):
        """
        Perform a backward pass through the NeuralODE
        joint_end_state is the final state, it contains the values of the
        adjoint, loss gradient wrt the parameters and state at the end time
        """
        if isinstance(self.func, Func):
            term = self.full_term_func
        elif isinstance(self.func, PDEFunc):
            term = self.full_term_pdefunc
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
            res['state_norm'] = jnp.linalg.norm(y_pred)
        if 'grad_init' in keys:
            to_grad = lambda y: self.solve(ts, y)
            res['grad_init'] = jax.jacfwd(to_grad)(y0).ys[-1, :, :]
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
        