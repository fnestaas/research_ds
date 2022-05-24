from models.Func import RegularFunc, PDEFunc
import jax.random as jrandom 
import jax.numpy as jnp 
import matplotlib.pyplot as plt

res = []
d = 100
width_size = d
depth = 2
seed = 10
skew = True

N_seeds = 50

for d in range(2, 202, 10):
    f1 = PDEFunc(d, width_size, depth, seed, skew=skew)
    f2 = RegularFunc(d, width_size, depth, seed=seed)

    r = 0
    for seed in range(N_seeds):
        key = jrandom.PRNGKey(seed)
        x = jrandom.normal(key, (d, ))
        n1 = jnp.linalg.norm(f1(None, x, None))
        n2 = jnp.linalg.norm(f2(None, x, None))
        r = r + n1 / n2
    res.append(r/N_seeds)

plt.plot(list(range(2, 202, 10)), res)
plt.show()
