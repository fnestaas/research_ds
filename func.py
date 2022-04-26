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
    L: float
    seed: int

    # TODO: check that the norm of the adjoint remains constant
    # TODO: try out a system where we add Bx + f0 to the solution, where B = anti-symmetric, learnable, f0 learnable const

    def __init__(self, d: int, L: float, width_size: int, depth: int, seed=0, **kwargs) -> None:
        super().__init__(**kwargs)

        self.d = d
        self.L = L
        self.seed = seed

        key = jrandom.PRNGKey(seed)
        k1, k2 = jrandom.split(key, 2)
        in_size = d
        self.init_nn = MLPWithParams(in_size, out_size=in_size, width_size=width_size, depth=depth, key=k1)        
        # grad_out = int((d + 1) * d / 2) # number of parameters for skew-symmetric matrix of shape (d+1, d+1)
        grad_out = d ** 2
        self.grad_nn = MLPWithParams(in_size, grad_out, width_size, depth, key=k2) # predicts gradient of f

        self.n_params = self.init_nn.n_params + self.grad_nn.n_params

    def __call__(self, ts, x, args):
        # integrate
        # z = jnp.concatenate([x, jnp.array([self.L])])
        z = x
        s = jnp.linspace(0, 1, 101)
        y = jax.vmap(self.integrand, in_axes=(None, 0))(z, s) # A(sx)x
        integral = jnp.trapz(y, x=s, dx=1e-2, axis=0) # TODO: redo this
        # integral = self.integrand(z, 1)

        assert integral.shape == (self.d, ), f'shape of integral is {integral.shape}'

        return integral + self.pred_init(x)

    def integrand(self, x, s):
        d = self.d
        out = self.grad_nn(s*x) # \nabla f(s*x)
        out = jnp.concatenate([out, jnp.zeros(d*d - out.shape[0])]) # make conformable to (d, d)-matrix
        out = jnp.reshape(out, (d, d))
        out = out - jnp.transpose(out)
        return out @ x

    def pred_init(self, x):
        # x is only used for shape. This prediction cannot depend on x!
        return self.init_nn(jnp.zeros(x.shape)).reshape(x.shape)

    def compute_adjoint(self, x, ts, end_state):
        saveat = diffrax.SaveAt(ts=ts[::-1]) # diffrax doesn't work otherwise
        y0 = jnp.concatenate([end_state, x.reshape((-1, ))])
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.term),
            diffrax.Tsit5(),
            t0=ts[-1],
            t1=ts[0],
            dt0=-(ts[1]-ts[0]),
            y0=end_state,
            stepsize_controller=diffrax.PIDController(),
            saveat=saveat,
        )
        return solution

    def term(self, t, state, args):
        n = self.d
        adjoint = state[1:n+1]
        x = state[n+1:].reshape((-1, n))
        N = x.shape[0]
        idx = int(t * N / 10)
        x = x[idx, :]

        adjoint_change = adjoint @ self.grad_nn(x).reshape((n, n))

        return jnp.concatenate([
            adjoint_change,
            jnp.zeros((N, ))
        ])


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