from models.NeuralODE import *
from models.nn_with_params import *
from models.Func import *
import equinox as eqx
import jax.numpy as jnp
import jax.nn as jnn

class NeuralODEClassifier(eqx.Module):
    input_layer: LinearWithParams
    output_layer: LinearWithParams
    activation: Callable
    node: NeuralODE
    func: Func

    def __init__(self, func, in_size: int, out_size: int, key, activation: Callable=jnn.sigmoid, to_track: List=['num_steps', 'state_norm', 'grad_init'],  **kwargs) -> None:
        super().__init__(**kwargs)
        self.func = func
        self.node = NeuralODE(func, to_track=to_track)
        self.input_layer = LinearWithParams(in_size, func.d, key=key)
        self.output_layer = LinearWithParams(func.d, out_size, key=key)
        self.activation = activation

    def __call__(self, ts, x, update=False):
        x = self.input_layer(x)
        x = self.node(ts, x, update=update)
        x = self.output_layer(x[-1, :]) # use x at final time
        return self.activation(x)