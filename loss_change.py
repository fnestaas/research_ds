import jax
import jax.numpy as jnp

def loss_change(model, grads, ts, ys, reduce=True, representative_sample=True):
    """
    compute the time derivative of the grads
    Parameters:
        model (NeuralODE): The model which contains the structure that grads should inherit
        grads (NeuralODE): gradients of a loss, potentially missing some parameters of a regular NeuralODE
        ts (jnp.array): time points on which to evaluate the time derivative
        ys (jnp.array): initial states at which to evaluate the NeuralODE
        reduce (boolean): Whether to reduce the output from dimension [batch_size, n_timesteps, dim_y, n_timesteps] 
                        to [batch_size, n_timesteps, dim_y]. This can be done because the solution at different times
                        is independent and therefore the derivative is 0 at these indices 
    """
    callable_grads = make_grads_callable(model, grads)
    if representative_sample:
        to_diff = lambda t: callable_grads(t, ys[0, 0, :], False)
    else:
        to_diff = lambda t: jax.vmap(callable_grads, in_axes=(None, 0, None))(t, ys[:, 0, :], False)
    if reduce:
        dLdt = jnp.sum(jax.jacrev(to_diff)(ts), axis=1)
    else:
        dLdt = jax.jacrev(to_diff)(ts)
    return dLdt

def make_grads_callable(model, grads):
    """
    the gradients are not callable since the parameters that should not be updated are set to None. This function sets
    the parameters that are None in grads and not None in model to the value they take in model.
    """
    return jax.tree_map(_fill_none, model, grads)

def _fill_none(m, g):
    if g is None and m is not None:
        return m 
    else:
        return g