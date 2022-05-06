import time
import jax 
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom 
import diffrax
import equinox as eqx
import matplotlib.pyplot as plt
import optax
import copy
from models.WeightDynamics import * 
from models.NeuralODE import *


seed = 0

d = 2
width = 10
depth = 2
key = jrandom.PRNGKey(seed)

b = GatedODE(d=d, width=width, depth=depth, key=key)
model = NeuralODE(b)

print(model.n_params)

params = model.get_params(as_dict=False)
print(len(params))
model.set_params(params, as_dict=False)
a = 0