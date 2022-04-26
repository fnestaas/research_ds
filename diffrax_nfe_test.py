import jax.numpy as jnp 
import joblib 
import matplotlib.pyplot as plt 

stats = joblib.load('outputs/state_norm.pkl')#joblib.load('outputs/stats.pkl')
key = 'grad_info' # 'num_steps' 
# N = len(stats[key])
N = len(stats)

# stat = [jnp.mean(jnp.abs(s)) for s in stats[key]]
stat = [jnp.var(s) for s in stats]

#stat[:200] = [s / jnp.sqrt(10) for s in stat[:200]]
#stat[200:] = [s / jnp.sqrt(100) for s in stat[200:]]


fig, ax = plt.subplots(1, 1)

ax.plot(
    jnp.arange(N), 
    stat,
    )

ax.set_xlabel('Epoch')
ax.set_ylabel(key)
ax.set_ylim([0, max(stat)*1.1])

plt.show()
