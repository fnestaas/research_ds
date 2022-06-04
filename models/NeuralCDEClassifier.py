from models.NeuralCDE import *
from models.nn_with_params import *
import equinox as eqx
import jax.numpy as jnp
import jax.nn as jnn
from jax import vmap, grad

class NeuralCDEClassifier(eqx.Module):
    output_layer: LinearWithParams
    activation: Callable
    ncde: NeuralCDE
    func: CDEFunc
    n_params: int
    use_out: bool

    def __init__(self, 
        func, 
        init_width: int, 
        init_depth:int, 
        out_size: int, 
        key, 
        activation: Callable=jnn.softmax, 
        to_track: List=['num_steps', 'state_norm', ], 
        rtol=1e-3, 
        atol=1e-6, 
        use_out=True, 
        input_layer=None, 
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.func = func
        self.ncde = NeuralCDE(
            width_size=init_width, 
            depth=init_depth,
            func=func, 
            d=func.d, 
            hidden_size=func.hidden_size, 
            to_track=to_track, 
            rtol=rtol, 
            atol=atol, 
            classify=False,
            key=key
        )
        self.output_layer = LinearWithParams(func.hidden_size, out_size, key=key)
        self.activation = activation
        self.n_params = self.ncde.n_params + self.output_layer.n_params # only care about this
        self.use_out = use_out

    def backward(self, ti, yi, loss_func, labels, N=100):
        new_t = jnp.linspace(ti[0], ti[-1], N)
        node_out = vmap(self.pred_partial, in_axes=(None, 0, None))(ti, yi, False)

        end_adjoint = grad(lambda y: loss_func(labels, vmap(self.pred_rest, in_axes=(None, 0))(ti, y), self))(node_out)

        joint_end_state = jnp.concatenate([end_adjoint, node_out], axis=-1)
        
        backward_pass = vmap(self.ncde.backward, in_axes=(None, 0))(new_t, joint_end_state)

        return backward_pass

    def __call__(self, ts, x, update=False):
        x = self.pred_partial(ts, x, update=update)
        return self.pred_rest(ts, x)
        

    def get_stats(self, which=None):
        return self.ncde.get_stats(which=which)

    def pred_partial(self, ts, x, update=False):
        return self.ncde(ts, x)[-1, :]
    
    def pred_rest(self, ts, x):
        if self.use_out:
            x = self.output_layer(x) 
        return self.activation(x)

    def get_params(self):
        return jnp.concatenate([self.ncde.get_params(), self.output_layer.get_params()])# ignore others

    def set_params(self, params):
        n1 = 0
        n2 = n1 + self.ncde.n_params
        self.ncde.set_params(params[n1:n2])
        self.output_layer.set_params(params[n2:])