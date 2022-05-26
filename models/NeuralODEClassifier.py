from models.NeuralODE import *
from models.nn_with_params import *
from models.Func import *
import equinox as eqx
import jax.numpy as jnp
import jax.nn as jnn
from jax import vmap, grad

class NeuralODEClassifier(eqx.Module):
    input_layer: LinearWithParams
    output_layer: LinearWithParams
    activation: Callable
    node: NeuralODE
    func: Func
    n_params: int
    use_out: bool

    def __init__(self, 
        func, 
        in_size: int, 
        out_size: int, 
        key, 
        activation: Callable=jnn.softmax, 
        to_track: List=['num_steps', 'state_norm', ], 
        rtol=1e-3, 
        atol=1e-6, 
        use_out=False, 
        input_layer=None, 
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.func = func
        self.node = NeuralODE(func, to_track=to_track, rtol=rtol, atol=atol)
        if input_layer is not None:
            self.input_layer = input_layer
        else:
            self.input_layer = LinearWithParams(in_size, func.d, key=key)
        self.output_layer = LinearWithParams(func.d, out_size, key=key)
        self.activation = activation
        self.n_params = self.node.n_params + self.input_layer.n_params + self.output_layer.n_params # only care about this
        self.use_out = use_out

    def backward(self, ti, yi, loss_func, labels, N=100):
        new_t = jnp.linspace(ti[0], ti[-1], N)
        node_out = vmap(self.pred_partial, in_axes=(None, 0, None))(ti, yi, False)

        end_adjoint = vmap(
            grad(lambda y: loss_func(labels, self.pred_rest(ti, y), self)), 
            in_axes=0
        )(node_out) 

        joint_end_state = jnp.concatenate([end_adjoint, node_out], axis=-1)
        
        backward_pass = vmap(self.node.backward, in_axes=(None, 0))(new_t, joint_end_state)

        return backward_pass

    def __call__(self, ts, x, update=False):
        x = self.pred_partial(ts, x, update=update)
        return self.pred_rest(ts, x)
        

    def get_stats(self, which=None):
        return self.node.get_stats(which=which)

    def pred_partial(self, ts, x, update=False):
        x = self.input_layer(x)
        return self.node(ts, x, update=update)[-1, :]
        # return jnp.mean(self.node(ts, x, update=update)[::2, :], axis=0)
    
    def pred_rest(self, ts, x):
        if self.use_out:
            x = self.output_layer(x) 
        return self.activation(x)

    def get_params(self):
        # return self.node.get_params()# ignore others
        return jnp.concatenate([self.input_layer.get_params(), self.node.get_params(), self.output_layer.get_params()])# ignore others

    def set_params(self, params):
        n1 = self.input_layer.n_params
        n2 = n1 + self.node.n_params
        self.input_layer.set_params(params[:n1])
        self.node.set_params(params[n1:n2])
        self.output_layer.set_params(params[n2:])