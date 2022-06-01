from xmlrpc.client import Boolean, boolean
import jax.numpy as jnp
import jax
import diffrax
import equinox as eqx
from numpy import diff
from models.NeuralODE import NeuralODE, StatTracker
from models.nn_with_params import *
from abc import abstractmethod
from equinox.nn.composed import _identity
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

class CDEFunc(eqx.Module):
    d: int 
    hidden_size: int

    @abstractmethod
    def get_params(self, as_dict=False):
        pass

    @abstractmethod
    def set_params(self, params, as_dict=False):
        pass


class CDEPDEFunc(CDEFunc):
    init_nn: LinearWithParams# MLPWithParams
    grad_nn: MLPWithParams
    d: int
    hidden_size: int
    n_params: int
    seed: int
    skew: bool # whether to predict using a skew-symmetric matrix
    integrate: bool

    def __init__(self, d: int, hidden_size: int,  width_size: int, depth: int, seed=0, skew=True, final_activation=_identity, integrate=False, **kwargs) -> None:
        super().__init__(d, hidden_size, **kwargs)

        self.d = d
        self.seed = seed

        key = jrandom.PRNGKey(seed)
        k1, k2 = jrandom.split(key, 2)

        in_size = hidden_size
        regular_out = d * hidden_size
        self.init_nn = LinearWithParams(1, regular_out, key=k1)# MLPWithParams(in_size, out_size=regular_out, width_size=width_size, depth=1, key=k1, final_activation=final_activation)   

        grad_out = hidden_size * regular_out
        self.grad_nn = MLPWithParams(in_size, grad_out, width_size, depth, key=k2, final_activation=final_activation) # predicts gradient of f

        params = self.grad_nn.get_params()
        k = grad_out * (width_size + 1) # the parameters which are in the last layer (+1 for bias)
        self.grad_nn.set_params(
            jnp.concatenate(
                [
                    params[:-k], 
                    # normalize by the product since this is involved in two matrix-vector products
                    params[-k:] / jnp.sqrt(d*(hidden_size - int(skew))) 
                ]
            )
        )

        params = self.init_nn.get_params()
        k = regular_out * (width_size + 1) # the parameters which are in the last layer (+1 for bias)
        self.init_nn.set_params(
            jnp.concatenate(
                [
                    params[:-k], 
                    # normalize since we multiply by a d-dimensional vector
                    # Empirically, it looks like we should also normalize by the hidden size
                    params[-k:] / jnp.sqrt(d*(hidden_size - int(skew)))
                ]
            )
        )
        
        self.n_params = self.init_nn.n_params + self.grad_nn.n_params
        self.skew = skew
        self.hidden_size = hidden_size
        self.integrate = integrate

    def __call__(self, ts, x, args):
        z = x
        if self.integrate:
            N = 100
            s = jnp.linspace(0, 1, N+1)
            y = jax.vmap(self.integrand, in_axes=(None, 0))(z, s) # A(sx)x
            integral = jnp.trapz(y, x=s, dx=1/N, axis=0) 
        else:
            tau = 1/2 # hyperparameter; how to approximate the integral best possible? Actually also good performance for small values (1/10 e.g.)
            integral = self.integrand(z, tau) 
        
        return integral + self.pred_init()

    def integrand(self, x, s):
        out = self.pred_mat(x, s) # (hidden_size, d, hidden_size)
        res = jnp.matmul(out, x) # (hidden_size, d)
        return res

    def pred_mat(self, x, s):
        d = self.d
        out = self.grad_nn(s*x) # \nabla f(s*x)
        out = jnp.reshape(out, (self.hidden_size, d, self.hidden_size))
        if self.skew:
            out = out - jnp.transpose(out, (2, 1, 0))
            out = out / jnp.sqrt(2) # should still be okay
        return out # (hidden_size, d, hidden_size)
        
    def pred_init(self):
        return self.init_nn(jnp.array([
            0 # could be time if desired
        ])).reshape((self.hidden_size, self.d))

    def get_params(self, as_dict=False):
        if as_dict:
            raise NotImplementedError
        return jnp.concatenate([self.init_nn.get_params(as_dict=as_dict), self.grad_nn.get_params(as_dict=as_dict)])

    def set_params(self, params, as_dict=False):
        if as_dict:
            raise NotImplementedError 
        n_init = self.init_nn.n_params
        self.init_nn.set_params(params[:n_init], as_dict=as_dict)
        self.grad_nn.set_params(params[n_init:], as_dict=as_dict)


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

    def get_params(self, as_dict=False):
        if as_dict:
            raise NotImplementedError
        return self.nn.get_params()

    def set_params(self, params, as_dict=False):
        if as_dict:
            raise NotImplementedError 
        self.nn.set_params(params)


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

    def __init__(self, d, width_size, depth, hidden_size, func, *, key, to_track=['num_steps', 'state_norm'], **kwargs):
        super().__init__(func, to_track=to_track, **kwargs)
        import warnings 
        warnings.warn(
            "NeuralCDE is currently implemented as a classifier. Would be better to make this a separate class."
        )
        ikey, fkey, lkey = jrandom.split(key, 3)
        self.initial = MLPWithParams(d, hidden_size, width_size, depth, key=ikey)
        self.linear = LinearWithParams(hidden_size, 1, key=lkey)

        # params = self.linear.get_params()
        # self.linear.set_params(params / jnp.sqrt(hidden_size))

        self.func = func
        self.stats = StatTracker(to_track)
        self.n_params = self.func.n_params + self.initial.n_params
        self.d = self.func.d

    def cde_backward_term(self, t, joint_state, args, cde_term):
        n = self.func.hidden_size
        adjoint = joint_state[:n]
        z = joint_state[-n:]

        dfdz = jax.jacfwd(lambda x: cde_term.vf(t, x, args)) # evaluate the vector field vf
        
        t1 = - adjoint @ dfdz(z)
        t2 = cde_term.vf(t, z, args)
        return jnp.concatenate([t1, t2])

    def backward(self, ts, coeffs, loss_func, label):
        # coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, joint_end_state)
        control = diffrax.CubicInterpolation(ts, coeffs)
        cde_term = diffrax.ControlTerm(self.func, control).to_ode() 
        term = diffrax.ODETerm(lambda t, joint_state, args: self.cde_backward_term(t, joint_state, args, cde_term))
        solver = diffrax.Tsit5()
        dt0 = -(ts[-1] - ts[0]) # integrating backwards in time

        _, output = self.solve(ts, coeffs)

        y_final = output.ys[-1, :]
        adjoint_final = jax.grad(lambda z: loss_func(label, self.pred_final(z)))(y_final)

        y0 = jnp.concatenate([adjoint_final, y_final])
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
            # max_steps=2**14,
            # adjoint=diffrax.BacksolveAdjoint(),
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
            max_steps=2**14,
        )
        if evolving_out:
            prediction = jax.vmap(lambda y: self.pred_final(y)[0])(solution.ys)
        else:
            (prediction,) = self.pred_final(solution.ys[-1])
        return prediction, solution

    def __call__(self, ts, coeffs, evolving_out=False, update=True):
        preds, sol = self.solve(ts, coeffs, evolving_out)
        if update:
            self.stats.update(self.compute_stats(sol))
        return preds

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

    def get_params(self, as_dict=False):
        if as_dict:
            raise NotImplementedError
        else:
            return jnp.concatenate([self.initial.get_params(), self.func.get_params()])

    def set_params(self, params, as_dict=False):
        if as_dict:
            raise NotImplementedError
        else:
            n1 = self.initial.n_params
            self.initial.set_params(params[:n1], as_dict=as_dict)
            self.func.set_params(params[n1:])
