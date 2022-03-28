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

a = jrandom.normal(key=jrandom.PRNGKey(0), shape=(2, 2))
print(a)

f = lambda x: jnp.matmul(jnp.transpose(x), jnp.matmul(a, x))[0, 0]

x = jnp.ones(shape=(2, 1))

print(jax.grad(f)(x))