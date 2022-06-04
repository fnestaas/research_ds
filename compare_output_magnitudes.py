from ast import keyword
from models.Func import RegularFunc, PDEFunc
from models.NeuralCDE import CDERegularFunc, CDEPDEFunc
import jax.random as jrandom 
import jax.numpy as jnp 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def fwd(x, seed, d, depth, name):
    fac = 4
    tau = 1 # does not impact magnitude
    if name == 'RegularFunc':
        return RegularFunc(d=d, width_size=fac*d, depth=depth, seed=seed)(None, x, None)
    elif name == 'PDEFunc':
        return PDEFunc(d=d, width_size=fac*d, depth=depth, seed=seed, integrate=False, skew=True)(None, x, None)
    elif name == 'CDERegularFunc': # TODO: d=1? We are taking the norm in the end
        return jnp.matmul(CDERegularFunc(d=d, hidden_size=d, width_size=fac*d, depth=depth, seed=seed)(None, x, None), x)
    elif name == 'CDEPDEFunc':
        return jnp.matmul(CDEPDEFunc(d=d, hidden_size=d, width_size=fac*d, depth=depth, seed=seed, integrate=False, tau=tau)(None, x, None), x)
    elif name == 'x':
        return x
    else:
        raise NotImplementedError

verbose = True
N_seeds = 3

ds = [5, 10, 20, 50]
depths = [5, 8, 10]

names = [
    'RegularFunc', 
    'PDEFunc', 
    # 'CDERegularFunc', 
    # 'CDEPDEFunc', 
    'x']

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

row = 1
col = 2

# ds, depths = np.meshgrid(ds[1:], depths[1:]) # artifacts

# fig, axs = plt.subplots(row, col, subplot_kw={'projection': '3d'})
fig, axs = plt.subplots(row, col*2)
for i in range(row):
    for j in range(col):
        if row > 1 and col > 1:
            ax = axs[i, j]
        else:
            ax = axs[col*i + j]
        name = names[col*i + j]
        n = len(depths)
        # ax.plot_surface(ds, depths, result[name][-n:, -n:], cmap=cm.coolwarm)
        im = ax.imshow(result[name])
        ax.set_title(name)
        ax.set_xlabel('depth')
        ax.set_ylabel('d')
        print(jnp.max(result[name]))
        # ax.set_zlim(0, 50)
        fig.colorbar(im, axs[col*i + j + col])

# to_plot = result['CDERegularFunc'] / result['CDEPDEFunc']
# fig, ax = plt.subplots(1, 1,)
# ax.imshow(to_plot)
# ax.set_yticks(np.arange(len(ds)), labels=ds)
# ax.set_xticks(np.arange(len(depths)), labels=depths)
# ax.set_ylabel('d')
# ax.set_xlabel('depth')
# for i in range(len(ds)):
#     for j in range(len(depths)):
#         text = ax.text(j, i, round(to_plot[i, j], 2),
# ha="center", va="center", color="w")
plt.show()