from abc import abstractmethod
import equinox as eqx
from models.WeightDynamics import * 
from models.nn_with_params import *
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

        t1 = - adjoint @ dfdz(x)
        t2 = adjoint @ dfdth(self.func.get_params(as_dict=False))
        t3 = self.func(t, x, args)

        return jnp.concatenate([t1, t2, t3]) # don't negate; diffrax does that

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

        t1 = - adjoint @ dfdz(x)
        # t1 = - adjoint @ self.func.pred_skew(x, 1) # This is skew symmetric and then the adjoint norm 
        # is constant;
        # which is further evidence that the variation of the adjoint arises from numerical errors in 
        # differentiation/integration
    
        # t2 = adjoint @ dfdth(self.func.get_params(as_dict=False))
        t3 = self.func(t, x, args)

        # return jnp.concatenate([t1, t2, t3])
        return jnp.concatenate([t1, t3])

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

class Func(eqx.Module):
    d: int 

    @abstractmethod
    def get_params(self, as_dict=False):
        pass

    @abstractmethod
    def set_params(self, params, as_dict=False):
        pass

class ODE2ODEFunc(Func):
    """
    Complete dynamics of the system; keeps track of how the weights and state evolve.
    """
    b: WeightDynamics
    f: DynX
    d: int
    n_params: int

    def __init__(self, b, f, **kwargs):
        super().__init__(b.d, **kwargs)
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

class PDEFunc(Func):
    init_nn: MLPWithParams
    grad_nn: MLPWithParams
    d: int
    n_params: int
    seed: int
    N: int # number of integration steps

    # TODO: check that the norm of the adjoint remains constant
    # TODO: try out a system where we add Bx + f0 to the solution, where B = anti-symmetric, learnable, f0 learnable const

    def __init__(self, d: int, width_size: int, depth: int, seed=0, N=100, **kwargs) -> None:
        super().__init__(d, **kwargs)

        self.d = d
        self.seed = seed

        key = jrandom.PRNGKey(seed)
        k1, k2 = jrandom.split(key, 2)
        in_size = d
        self.init_nn = MLPWithParams(in_size, out_size=in_size, width_size=width_size, depth=depth, key=k1)        
        grad_out = int((d - 1) * d / 2) # number of parameters for skew-symmetric matrix of shape (d, d)
        # grad_out = d ** 2
        self.grad_nn = MLPWithParams(in_size, grad_out, width_size, depth, key=k2) # predicts gradient of f

        self.n_params = self.init_nn.n_params + self.grad_nn.n_params
        self.N = N

    def __call__(self, ts, x, args):
        # integrate
        # z = jnp.concatenate([x, jnp.array([self.L])])
        z = x
        s = jnp.linspace(0, 1, self.N+1)
        y = jax.vmap(self.integrand, in_axes=(None, 0))(z, s) # A(sx)x
        integral = jnp.trapz(y, x=s, dx=1/self.N, axis=0) 
        # integral = self.integrand(z, 1) # TODO: for some reason this does not work well with adjoint

        assert integral.shape == (self.d, ), f'shape of integral is {integral.shape}'

        return integral + self.pred_init()

    def integrand(self, x, s):
        out = self.pred_skew(x, s)
        return out @ x

    def pred_skew(self, x, s):
        d = self.d
        out = self.grad_nn(s*x) # \nabla f(s*x)
        out = jnp.concatenate([out, jnp.zeros(d*d - out.shape[0])]) # make conformable to (d, d)-matrix
        out = jnp.reshape(out, (d, d))
        out = out - jnp.transpose(out)
        return out

    def pred_init(self):
        return self.init_nn(jnp.zeros((self.d, ))).reshape((self.d, ))

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

