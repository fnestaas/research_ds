import jax.numpy as jnp 
import joblib 
import matplotlib.pyplot as plt 

keys = [
    'adjoint_norm_ODE2ODE_unregularized_length_strategy', 
    'adjoint_norm_manual_grad', 
    'adjoint_norm_autodiff'
    ] 
fig, ax = plt.subplots(1, 1)
y_max = 0
for key in keys:
    stats = joblib.load(f'outputs/{key}.pkl')#joblib.load('outputs/stats.pkl') 
    # N = len(stats)

    stat = [jnp.mean(jnp.var(s, axis=-1)) for s in stats[60:]]
    N = len(stat)

    #stat[:200] = [s / jnp.sqrt(10) for s in stat[:200]]
    #stat[200:] = [s / jnp.sqrt(100) for s in stat[200:]]

    ax.plot(
        jnp.arange(N), 
        stat,
        label=key
        )
    y_max = max([max(stat)*1.1, y_max])

ax.set_xlabel('Epoch')
ax.legend(keys)
ax.set_ylim([0, y_max])

plt.show()
