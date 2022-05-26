from abc import abstractmethod
import equinox as eqx
from models.WeightDynamics import * 
from models.nn_with_params import *
import diffrax
from equinox.nn.composed import _identity


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
            if self.n_params is not None:
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
    skew: bool # whether to predict using a skew-symmetric matrix
    integrate: bool # whether to predict by integrating
    # efficient: bool

    # TODO: check that the norm of the adjoint remains constant
    # TODO: try out a system where we add Bx + f0 to the solution, where B = anti-symmetric, learnable, f0 learnable const

    def __init__(self, d: int, width_size: int, depth: int, seed=0, N=100, skew=True, integrate=True, final_activation=_identity, **kwargs) -> None:
        super().__init__(d, **kwargs)

        self.d = d
        self.seed = seed

        key = jrandom.PRNGKey(seed)
        k1, k2 = jrandom.split(key, 2)
        in_size = d
        self.init_nn = MLPWithParams(in_size, out_size=in_size, width_size=width_size, depth=depth, key=k1)   
        grad_out = d ** 2
        self.grad_nn = MLPWithParams(in_size, grad_out, width_size, depth, key=k2, final_activation=final_activation) # predicts gradient of f
        # we need to be careful when initializing the grad_nn for the distribution of the output to be the same as
        # if we had not done matrix multiplication. Below, we make take care of this
        params = self.grad_nn.get_params() 
        k = grad_out * (width_size + 1) # the parameters which are in the last layer (+1 for bias)
        self.grad_nn.set_params(jnp.concatenate([params[:-k], params[-k:] / jnp.sqrt(d - int(skew))]))
        # self.efficient = efficient
        self.n_params = self.init_nn.n_params + self.grad_nn.n_params
        self.N = N
        self.skew = skew
        self.integrate = integrate

    def __call__(self, ts, x, args):
        # integrate
        # z = jnp.concatenate([x, jnp.array([self.L])])
        z = x
        if self.integrate:
            s = jnp.linspace(0, 1, self.N+1)
            y = jax.vmap(self.integrand, in_axes=(None, 0))(z, s) # A(sx)x
            integral = jnp.trapz(y, x=s, dx=1/self.N, axis=0) 
        else:
            integral = self.integrand(z, 1) 

        return integral + self.pred_init()

    def integrand(self, x, s):
        out = self.pred_mat(x, s)
        return out @ x

    def pred_mat(self, x, s):
        d = self.d
        out = self.grad_nn(s*x) # \nabla f(s*x)
        # if self.efficient:
            # out = jnp.tril(out, -1)
            # res = jnp.zeros((d, d))
            # k = 0
            # for i in range(d-1):
            #     for j in range(i+1, d):
            #         res[i, j] = res.at[i, j].set(out.at[k])
            #         k = k + 1
            # out = out.at[jnp.triu_indices(n=d, k=1)].set(out)
        # else:
            # out = jnp.concatenate([out, jnp.zeros(d*d - out.shape[0])]) # make conformable to (d, d)-matrix
        out = jnp.reshape(out, (d, d))
        if self.skew:
            out = out - jnp.transpose(out) 
            out = out / jnp.sqrt(2)
        return out

    def pred_skew(self, x, s):
        assert self.skew
        return self.pred_mat(x, s)
        

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

class RegularFunc(Func):
    nn: MLPWithParams
    d: int
    n_params: int
    seed: int

    def __init__(self, d: int, width_size: int, depth: int, seed=0, final_activation=_identity, **kwargs) -> None:
        super().__init__(d, **kwargs)

        self.d = d
        self.seed = seed

        key = jrandom.PRNGKey(seed)
        in_size = d
        self.nn = MLPWithParams(in_size, out_size=in_size, width_size=width_size, depth=depth, key=key, final_activation=final_activation)        

        self.n_params = self.nn.n_params

    def __call__(self, ts, x, args):
        return self.nn(x)

    def get_params(self, as_dict=False):
        if as_dict:
            raise NotImplementedError
        return self.nn.get_params()

    def set_params(self, params, as_dict=False):
        if as_dict:
            raise NotImplementedError 
        self.nn.set_params(params)

class PWConstFunc(Func):
    nn1: MLPWithParams
    nn2: MLPWithParams
    init_nn: MLPWithParams
    d: int
    n_params: int
    seed: int

    def __init__(self, d: int, width_size: int, depth: int, seed=0, final_activation=_identity, **kwargs) -> None:
        super().__init__(d, **kwargs)

        self.d = d
        self.seed = seed

        key = jrandom.PRNGKey(seed)
        in_size = d
        out_size = d*d
        self.nn1 = MLPWithParams(in_size=in_size, out_size=out_size, width_size=width_size, depth=depth, key=key, final_activation=final_activation)  
        self.nn2 = MLPWithParams(out_size, out_size=out_size, width_size=width_size, depth=2, key=key, final_activation=final_activation, activation=lambda x: jnp.float32(x > 0))  
        self.init_nn = MLPWithParams(in_size=in_size, out_size=in_size, width_size=width_size, depth=depth, key=key, final_activation=final_activation) 
        self.n_params = self.nn1.n_params + self.nn2.n_params

    def __call__(self, ts, x, args):
        d = self.d
        A = self.get_A(ts, x)
        return A @ x + self.init_nn(jnp.zeros(x.shape))
    
    def get_A(self, ts, x):
        d = self.d
        A = jnp.reshape(self.nn1(x), (d, d)) # some matrix
        A = self.nn2(jnp.reshape(A, (-1, )))
        A = jnp.reshape(A, (d, d)) # pw const matrix
        return (A - jnp.transpose(A))

    def get_params(self, as_dict=False):
        if as_dict:
            raise NotImplementedError
        return jnp.concatenate([self.nn1.get_params(as_dict=as_dict), self.nn2.get_params(as_dict=as_dict), self.init_nn.get_params(as_dict=as_dict)])

    def set_params(self, params, as_dict=False):
        if as_dict:
            raise NotImplementedError 
        n1 = self.nn1.n_params 
        n2 = n1+self.init_nn.n_params
        self.nn1.set_params(params[:n1], as_dict=as_dict)
        self.nn2.set_params(params[n1:n2], as_dict=as_dict)
        self.init_nn.set_params(params[n2:], as_dict=as_dict)
        