from xmlrpc.client import boolean
import jax.numpy as jnp
import jax
import diffrax
import equinox as eqx
from models.NeuralODE import NeuralODE, StatTracker
from models.nn_with_params import *
from models.CDEFunc import *
from abc import abstractmethod
from equinox.nn.composed import _identity
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

class NeuralCDE(NeuralODE):
    """
    Based on https://docs.kidger.site/diffrax/examples/neural_cde/
    Modified to fit into our implementation this far. 
    """
    initial: eqx.nn.MLP
    func: CDEFunc
    linear: LinearWithParams
    d: int
    n_params: int 
    stats: StatTracker
    classify: boolean

    def __init__(
        self, 
        d: int, 
        width_size: int, 
        depth: int, 
        hidden_size: int, 
        func: CDEFunc, 
        to_track=['num_steps', 'state_norm'], # stats to keep track of
        classify=True, 
        *, 
        key, 
        **kwargs
    ):
        super().__init__(func, to_track=to_track, **kwargs)
        ikey, fkey, lkey = jrandom.split(key, 3)
        self.initial = MLPWithParams(d, hidden_size, width_size, depth, key=ikey)
        self.linear = LinearWithParams(hidden_size, 1, key=lkey)

        self.func = func
        self.stats = StatTracker(to_track)
        self.n_params = self.func.n_params + self.initial.n_params
        self.d = self.func.d
        self.classify = classify 

    def cde_backward_term(self, t, joint_state, args, cde_term):
        n = self.func.hidden_size
        adjoint = joint_state[:n]
        z = joint_state[-n:]

        dfdz = jax.jacfwd(lambda x: cde_term.vf(t, x, args)) # evaluate the vector field vf
        
        t1 = - adjoint @ dfdz(z)
        t2 = cde_term.vf(t, z, args)
        return jnp.concatenate([t1, t2])

    def backward(self, ts, coeffs, loss_func, label):
        control = diffrax.CubicInterpolation(ts, coeffs)
        cde_term = diffrax.ControlTerm(self.func, control).to_ode() 
        term = diffrax.ODETerm(lambda t, joint_state, args: self.cde_backward_term(t, joint_state, args, cde_term))
        solver = diffrax.Tsit5()
        dt0 = -(ts[-1] - ts[0]) # integrating backwards in time

        # first solve since we need the final state to do the backward pass
        if self.classify:
            _, output = self.solve(ts, coeffs)
            
        else:
            output = self.solve(ts, coeffs, update=False)
        y_final = output.ys[-1, :]

        # compute the final value of the adjont
        adjoint_final = jax.grad(lambda z: loss_func(label, self.pred_final(z)))(y_final)

        y0 = jnp.concatenate([adjoint_final, y_final]) # initial state for backwards solving
        saveat = diffrax.SaveAt(ts=ts[::-1]) # diffrax doesn't work otherwise
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=ts[-1],
            t1=ts[0],
            dt0=dt0,
            y0=y0,
            stepsize_controller=diffrax.PIDController(),
            saveat=saveat,
        )
        return solution

    def pred_final(self, y):
        return jnn.sigmoid(self.linear(y))

    def solve(self, ts, coeffs, evolving_out=False):
        # Each sample of data consists of some timestamps `ts`, and some `coeffs`
        # parameterising a control path. These are used to produce a continuous-time
        # input path `control`.
        control = diffrax.CubicInterpolation(ts, coeffs)
        term = diffrax.ControlTerm(self.func, control).to_ode()
        solver = diffrax.Tsit5()
        dt0 = (ts[-1]-ts[0])
        y0 = self.initial(control.evaluate(ts[0]))
        if evolving_out:
            saveat = diffrax.SaveAt(ts=ts)
        else:
            saveat = diffrax.SaveAt(t1=True)
        solution = diffrax.diffeqsolve(
            term,
            solver,
            ts[0],
            ts[-1],
            dt0,
            y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=saveat,
            max_steps=2**15,
        )
        if self.classify:
            if evolving_out:
                prediction = jax.vmap(lambda y: self.pred_final(y)[0])(solution.ys)
            else:
                (prediction,) = self.pred_final(solution.ys[-1])
            return prediction, solution
        else:
            return solution

    def __call__(self, ts, coeffs, update=False, evolving_out=False):
        if self.classify:
            preds, sol = self.solve(ts, coeffs, evolving_out)
        else:
            sol = self.solve(ts, coeffs, evolving_out)
        if update:
            self.stats.update(self.compute_stats(sol))
        return preds if self.classify else sol.ys

    def compute_stats(self, solution):
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
        return res

    def get_params(self):
        return jnp.concatenate([self.initial.get_params(), self.func.get_params()])

    def set_params(self, params):
        n1 = self.initial.n_params
        self.initial.set_params(params[:n1])
        self.func.set_params(params[n1:])
