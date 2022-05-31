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
                    params[-k:] / jnp.sqrt(d)
                ]
            )
        )
        
        self.n_params = self.init_nn.n_params + self.grad_nn.n_params
        self.skew = skew
        self.hidden_size = hidden_size
        self.integrate = integrate

    def __call__(self, ts, x, args):
        # integrate
        # z = jnp.concatenate([x, jnp.array([self.L])])
        z = x
        if self.integrate:
            N = 10
            s = jnp.linspace(0, 1, N+1)
            y = jax.vmap(self.integrand, in_axes=(None, 0))(z, s) # A(sx)x
            integral = jnp.trapz(y, x=s, dx=1/N, axis=0) 
        else:
            integral = self.integrand(z, 1) 

        return integral + self.pred_init()

    def integrand(self, x, s):
        out = self.pred_mat(x, s) # (hidden_size, d, hidden_size)
        # perm = jnp.transpose(out, (1, 2, 0)) # (hidden_size, d, hidden_size)
        perm = out
        res = jnp.matmul(perm, x) # (hidden_size, d)
        return res

    def pred_mat(self, x, s):
        d = self.d
        out = self.grad_nn(s*x) # \nabla f(s*x)
        # out = jnp.reshape(out, (self.hidden_size, self.hidden_size, d))
        out = jnp.reshape(out, (self.hidden_size, d, self.hidden_size))
        if self.skew:
            # out = out - jnp.transpose(out, (1, 0, 2)) 
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

    def cde_backward_term(self, t, joint_state, args):
        n = self.func.d
        adjoint = joint_state[:n]
        x = joint_state[-n:]

        dfdz = jax.jacfwd(lambda z: self.func(t, z, args))
        
        t1 = - adjoint @ dfdz(x)
        t3 = self.func(t, x, args)
        return jnp.concatenate([t1, t3])

    def backward(self, ts, coeffs):
        # coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, joint_end_state)
        control = diffrax.CubicInterpolation(ts, coeffs)
        term = diffrax.ControlTerm(self.cde_backward_term, control).to_ode()
        solver = diffrax.Tsit5()
        dt0 = -(ts[-1] - ts[0]) # integrating backwards in time
        y0 = self.initial(control.evaluate(ts[0]))
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

    def solve(self, ts, coeffs, evolving_out=False):
        # Each sample of data consists of some timestamps `ts`, and some `coeffs`
        # parameterising a control path. These are used to produce a continuous-time
        # input path `control`.
        # assert y0.shape[-1] == self.d + 1, f'Bad shape {y0.shape}, NeuralCDE has {self.d=}. Did you forget time as a channel (0th dimension)?'
        # coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, y0) # hermite coeffs have to be computed on ts and ys of the same length; one y per t
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
            prediction = jax.vmap(lambda y: jnn.sigmoid(self.linear(y))[0])(solution.ys)
        else:
            (prediction,) = jnn.sigmoid(self.linear(solution.ys[-1]))
        return prediction, solution

    def __call__(self, ts, coeffs, evolving_out=False, update=True):
        preds, sol = self.solve(ts, coeffs, evolving_out)
        if update:
            self.stats.update(self.compute_stats(sol))
        return preds

    # def track_mode(self):
    #     self.update = True 
    
    # def eval_mode(self):
    #     self.update = False

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
