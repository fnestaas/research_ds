from ast import keyword
from models.Func import RegularFunc, PDEFunc
from models.NeuralCDE import CDERegularFunc, CDEPDEFunc
import jax.random as jrandom 
import jax.numpy as jnp 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def fwd(x, seed, d, depth, name):
    if name == 'RegularFunc':
        return RegularFunc(d=d, width_size=d, depth=depth, seed=seed)(None, x, None)
    elif name == 'PDEFunc':
        return PDEFunc(d=d, width_size=d, depth=depth, seed=seed, integrate=True)(None, x, None)
    elif name == 'CDERegularFunc': # TODO: d=1? We are taking the norm in the end
        return jnp.matmul(CDERegularFunc(d=d, hidden_size=d, width_size=d, depth=depth, seed=seed)(None, x, None), x)
    elif name == 'CDEPDEFunc':
        return jnp.matmul(CDEPDEFunc(d=d, hidden_size=d, width_size=d, depth=depth, seed=seed, integrate=True)(None, x, None), x)
    elif name == 'x':
        return x
    else:
        raise NotImplementedError

verbose = True
N_seeds = 3

ds = [2, 10, 50, 100]
depths = [2, 4, 10, 20]

names = ['RegularFunc', 'PDEFunc', 'CDERegularFunc', 'CDEPDEFunc', 'x']

result = {
    name: np.zeros((len(ds), len(depths))) for name in names
}


for i, d in enumerate(ds):
    for j, depth in enumerate(depths):
        if verbose: print(f'{i=}, {j=}')
        for seed in range(N_seeds):
            key = jrandom.PRNGKey(seed) 
            x = jrandom.normal(key=key, shape=(d, ))
            # check the magnitude of func(x)
            for name in names:
                result[name][i, j] += np.linalg.norm(fwd(x, seed, d, depth, name)) / N_seeds

fig, axs = plt.subplots(2, 3, subplot_kw={'projection': '3d'})
ds, depths = np.meshgrid(ds, depths)
for i in range(2):
    for j in range(3):
        ax = axs[i, j]
        name = names[2*i + j]
        ax.plot_surface(ds, depths, result[name], cmap=cm.coolwarm)
        ax.set_title(name)
        ax.set_xlabel('depth')
        ax.set_ylabel('d')

plt.show()