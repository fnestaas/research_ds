"""
ddx (A(x) x)_{i, j} = A_{i, j} (x) + \sum_k ddx (A_{i, k}) x_k

We are interested in the symmetric part of this, which is the sum of two symmetric matrices. 
How does the Frobenius norm of these matrices compare?
"""

import jax.numpy as jnp
import numpy as np
from models.Func import PDEFunc, RegularFunc # check only the func derivative
from jax import jacfwd 
import jax.random as jrandom
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle 

CHECK_BOTH = True 
CHECK_B = False
TAKE_LOG = True
TAKE_RATIO = False
SYM = True
N_runs = 1

def frobenius(A):
    return jnp.sum(jnp.square(A))

def get_diff(f, y):
    return jacfwd(lambda z: f(None, z, None))(y)

def sym_diff(f, y):
    diff = get_diff(f, y)
    result = 1/2 * (diff + jnp.transpose(diff))
    return result 

def get_mat(f, x):
    """
    get the matrix A at input x such that
    f(x) = f(0) + A(x)x
    """
    return f.pred_mat(x, 1)

def get_sym_frob(func, x, eps=1e-6, log=False, sym=SYM):
    diff = get_diff(func, x)
    if SYM:
        sd = 1/2 * (diff + jnp.transpose(diff))
    else:
        sd = 1/2 * (diff - jnp.transpose(diff))
    if log:
        return jnp.log(frobenius(sd) + eps) - 2*jnp.log(func.d) - jnp.log(frobenius(diff - sd) + eps) * int(TAKE_RATIO)
    else:
        return frobenius(sd) / func.d**2 / (frobenius(diff - sd) if TAKE_RATIO else 1) 

def mean_ratio(func, x):
    B = get_mat(func, x)
    B = 1/2 * (B + jnp.transpose(B))
    diff = sym_diff(func, x) 
    return jnp.array([frobenius(B), frobenius(diff - B), frobenius(diff)])

ds = [10, 50, 100, 150, 200]
depths = [5, 10, 20, 30, 50] # the deeper the more asymmetric, but also slower. For some reason different for regfunc
grid_skew = np.zeros((len(ds), len(depths)))
grid_any = np.zeros((len(ds), len(depths)))
grid_reg = np.zeros((len(ds), len(depths)))

if CHECK_BOTH:
    # check difference between skew and non-skew
    for i, d in enumerate(ds):
        for j, depth in enumerate(depths):
            rs = {True: [], False: [], 'reg': []}
            for seed in range(N_runs):
                key = jrandom.PRNGKey(seed) 
                x = jrandom.normal(key=key, shape=(d, ))
                for skew in [True, False]:
                    func = PDEFunc(d=d, seed=seed, width_size=d, depth=depth, skew=skew, integrate=False)
                    rs[skew].append(get_sym_frob(func, x, log=TAKE_LOG))
                    del func
                func = RegularFunc(d=d, seed=seed, width_size=d, depth=depth)
                rs['reg'].append(get_sym_frob(func, x, log=TAKE_LOG))
                del func
                
            rs[True] = jnp.mean(jnp.array(rs[True]))
            rs[False] = jnp.mean(jnp.array(rs[False]))
            rs['reg'] = jnp.mean(jnp.array(rs['reg']))
            grid_skew[i, j] = rs[True] 
            grid_any[i, j] = rs[False] 
            grid_reg[i, j] = rs['reg']
    
n_plots = 6
fig, axs = plt.subplots(1, n_plots, subplot_kw={"projection": "3d"})
poss = [None]*n_plots
ds, depths = np.meshgrid(ds, depths)
poss[0] = axs[0].plot_surface(ds, depths, grid_skew, cmap=cm.coolwarm)
axs[0].set_title('skew symmetric')
poss[1] = axs[1].plot_surface(ds, depths, grid_any, cmap=cm.coolwarm)
axs[1].set_title('any matrix')
poss[2] = axs[2].plot_surface(ds, depths, grid_skew/grid_any if not TAKE_LOG else grid_skew - grid_any, cmap=cm.coolwarm)
axs[2].set_title('ratio skew / any')
poss[3] = axs[3].plot_surface(ds, depths, grid_reg, cmap=cm.coolwarm)
axs[3].set_title('RegularFunc')
poss[4] = axs[4].plot_surface(ds, depths, grid_skew/grid_reg if not TAKE_LOG else grid_skew - grid_reg, cmap=cm.coolwarm)
axs[4].set_title('ratio skew / RegularFunc')
poss[5] = axs[5].plot_surface(ds, depths, grid_any/grid_reg if not TAKE_LOG else grid_any - grid_reg, cmap=cm.coolwarm)
axs[5].set_title('ratio any / RegularFunc')

for ax, pos in zip(axs, poss):
    ax.set_xlabel('depth')
    ax.set_ylabel('d')
    # fig.colorbar(pos, shrink=0.5, aspect=5)

plt.show()

with open('grid_skew.pkl', 'wb') as handle:
    pickle.dump(grid_skew, handle)

with open('grid_any.pkl', 'wb') as handle:
    pickle.dump(grid_any, handle)


if CHECK_B:
    # check how important B_{i, j} (x) is for the derivative magnitude
    skew = False
    rs = []
    for seed in range(N_runs):
        key = jrandom.PRNGKey(seed) 
        func = PDEFunc(d=d, seed=seed, width_size=d, depth=depth, skew=skew, integrate=False)
        x = jrandom.normal(key=key, shape=(d, ))
        rs.append(mean_ratio(func, x))
        del func

    rs = jnp.array(rs)

    rs = jnp.mean(rs, axis=0)

    print(f'\nmatrix, sum, total: {rs}\n')
