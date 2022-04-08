from copy import deepcopy
import jax 
import jax.numpy as jnp
from numpy import isin 
from loss_change import make_grads_callable, _fill_none
from copy import deepcopy

def loss_change_other_very_wrong(y, t, loss_func, grad, model):
    """
    Compute a of shape (bs, t, y)
    Compute func of shape (bs, t, y, thetas)
    for each (b, t, y) there should be a func which contains the gradients of the params to specify that func

    func_bty = jax.grad(y_pred[b,t,y]).func
    """
    y_pred = jax.vmap(model, in_axes=(None, 0, None))(t, y[:, 0, :], False)
    # callable_grad = make_grads_callable(model, grad)

    raise NotImplementedError # still have to fix this
    # to_map = lambda x: jax.vmap(, in_axes=(None, 0, None))(t, x, None)
    func = jax.vmap(to_map, in_axes=0)(y_pred)

    to_diff = lambda x: loss_func(x, y)
    a = jax.grad(to_diff)(y_pred) # notation adopted from NeuralODE paper
    return #jnp.sum(a * func, axis=-1) # jnp.tensordot or something

def _setter(f, t):
    if f is not None:
        return t 
    return None

def make_func_callable(model, func):
    return jax.tree_map(_fill_none, model, func)

def set_func_with_params(model, func, theta, template):
    t = redict([theta[i] for i in range(theta.shape[0])], template)
    set_params = func.set_params(t)
    return make_func_callable(set_params, model)

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
    theta_dict = make_grads_callable(model, grad).func.get_params()
    theta_list = undict(theta_dict)
    theta_list_copy = deepcopy(theta_list)

    func_helper_helper = lambda theta, x: jax.vmap(set_func_with_params(model, grad.func, theta, theta_dict), in_axes=(None, 0, None))(t, x, None)
    func_helper = lambda theta: jax.vmap(func_helper_helper, in_axes=(None, 0))(theta, y)
    func_grad = jax.jacrev(func_helper)

    theta_list_ = [jnp.reshape(i, (-1, )) for i in theta_list_copy]
    theta_ = jnp.concatenate(theta_list_, axis=0)
    test = redict(theta_.tolist(), theta_dict)
    res = func_grad(theta_)
    return res

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

