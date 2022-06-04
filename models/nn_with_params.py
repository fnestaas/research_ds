from equinox.nn import Linear, MLP
from equinox.nn.composed import Callable, List, _identity
import jax 
import jax.random as jrandom
import jax.nn as jnn
import jax.numpy as jnp

class LinearWithParams(Linear):
    n_params: int 

    def __init__(self, in_features: int, out_features: int, use_bias: bool = True, *, key: "jax.random.PRNGKey"):
        super().__init__(in_features, out_features, use_bias, key=key)
        self.n_params = (in_features + int(use_bias)) * out_features 


    def get_params(self):
        if self.bias is not None:
            return jnp.concatenate([self.bias, self.weight.reshape((-1, ))], axis=0) # return a single vector of parameters
        else:
            return self.weight.reshape((-1, ))

    def set_params(self, params):
        if self.n_params is not None:
            assert len(params) == self.n_params
        if self.bias is not None:
            bias = params[:self.out_features]
            weight = params[self.out_features:].reshape(self.weight.shape)
        else:
            weight = params.reshape(self.weight.shape)
        
        assert weight.shape == self.weight.shape, f'{weight.shape=}, {self.weight.shape=}'
        if self.bias is not None:
            object.__setattr__(self, 'bias', bias)
        object.__setattr__(self, 'weight', weight)

class MLPWithParams(MLP):
    layers: List[LinearWithParams]
    n_params: int
    in_size: int 
    out_size: int
    width_size: int 
    depth: int
    activation: Callable

    def __init__(self, in_size: int, out_size: int, width_size: int, depth: int, activation: Callable = jnn.relu, final_activation: Callable = _identity, *, key: "jax.random.PRNGKey", **kwargs):
        super().__init__(in_size, out_size, width_size, depth, activation, final_activation, key=key, **kwargs)
        
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.activation = activation

        keys = jrandom.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(LinearWithParams(in_size, out_size, key=keys[0]))
        else:
            layers.append(LinearWithParams(in_size, width_size, key=keys[0]))
            for i in range(depth - 1):
                layers.append(LinearWithParams(width_size, width_size, key=keys[i + 1]))
            layers.append(LinearWithParams(width_size, out_size, key=keys[-1]))
        self.layers = layers
        self.n_params = int(sum([l.n_params for l in layers]))

    def get_params(self, ):
        return jnp.concatenate([l.get_params() for l in self.layers], axis=0)

    def set_params(self, params):
        counter = 0
        for l in self.layers:
            layer_size = l.n_params
            p = params[counter:counter+layer_size]
            l.set_params(p)
            counter = counter + layer_size
