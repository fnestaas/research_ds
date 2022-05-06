from copy import deepcopy
import jax 
import jax.numpy as jnp
from numpy import save
from loss_change import make_grads_callable, _fill_none
from copy import deepcopy
import diffrax
from models.NeuralODE import Func

class AdjointSolver():
    func: Func 

    def __init__(self, func, **kwargs) -> None:
        b = func.b
        f = func.f
        self.func = Func(b, f, **kwargs)
        self.func.set_params(func.get_params())

    def solve(self, ts, loss_func, y, y_pred):
        to_diff = lambda x: loss_func(x, y)
        y0 = jax.grad(to_diff)(y_pred)
        arg0 = diffrax.ODETerm(self.func) # TODO: this is a bug; should be derivative wrt state!
        arg1 = diffrax.Tsit5()
        t0=ts[-1]
        t1=ts[0] # solve backwards in time
        dt0=ts[0] - ts[1]
        stepsize_controller=diffrax.PIDController()
        saveat=diffrax.SaveAt(ts=ts[::-1])
        solution = diffrax.diffeqsolve(
            arg0, 
            arg1, 
            y0=y0,
            t0=t0,
            t1=t1,
            dt0=dt0,
            stepsize_controller=stepsize_controller,
            saveat=saveat,  
            )
        return solution.ys

def make_func_callable(model, func):
    return jax.tree_map(_fill_none, model, func)

def set_func_with_params(func, theta):
    # t = redict([theta[i] for i in range(theta.shape[0])], template)
    #func_ = deepcopy(func)
    func.set_params(theta, as_dict=False)
    return func

def undict(d, structured=False, rm_none=True):
    """
    Make a list from a nested dict
    """
    if not isinstance(d, dict):
        if isinstance(d, list):
            return d 
        if rm_none and d is None:
            return []
        return [d]
    res = [] 
    if not structured:
        for v in d.values():
            res = res + undict(v)
    else:
        for v in d.values():
            res.append(undict(v, structured=True))
    return res

def redict(l, template):
    """
    put the elements of the list l into the nested format of template
    """
    if isinstance(template, dict):
        if isinstance(l, list):
            return {k: redict(l[i], v) for i, (k, v) in enumerate(template.items())}
        else:
            return {k: redict(l, v) for i, (k, v) in enumerate(template.items())}
    else:
        if isinstance(l, list): 
            return jnp.array(l)
        else:
            return l

def loss_change_other(y, t, loss_func, grad, model):
    callable_func = deepcopy(model.func) # deepcopy(make_func_callable(model.func, grad.func))
    func_helper_helper = lambda th, x: jax.vmap(set_func_with_params(callable_func, th), in_axes=(None, 0, None))(t, x, None)
    func_helper = lambda th: jax.vmap(func_helper_helper, in_axes=(None, 0))(th, y)
    func_grad = jax.jacrev(func_helper) # grad of func wrt theta

    theta = callable_func.get_params()
    f_grd = func_grad(theta)

    y_pred = jax.vmap(model, in_axes=(None, 0, None))(t, y[:, 0, :], False)
    to_diff = lambda x: loss_func(x, y)
    # a_ = lambda y_: jax.grad(to_diff)(y_) # notation adopted from NeuralODE paper
    # a = a_(jax.vmap(model, in_axes=(None, 0, None))(t, y[:, 0, :], False))

    # # to_diff = lambda t_: loss_func(jax.vmap(model, in_axes=(None, 0, None))(t_, y[:, 0, :], False), y)
    # # a = jax.grad(to_diff)(t)

    a_solver = AdjointSolver(callable_func)
    a = a_solver.solve(t, loss_func, y, y_pred)

    a_extended = jnp.repeat(jnp.expand_dims(a, axis=-1), f_grd.shape[-1], axis=-1)
    res = a_extended * f_grd
    return jnp.sum(res, axis=-2)

test = False
if test:
    d = {
        'a': {
            'b': [1, 2, 3], 
            'c': [5, 6, 7], 
        }, 
        #'d': [9]
    }

    l = undict(d, structured=True)
    print(l)
    print(redict(l, d))

