import jax.numpy as jnp 
import joblib 
import matplotlib.pyplot as plt 

stats = joblib.load('stats.pkl')
key = 'grad_init' # 'num_steps' 
N = len(stats[key])

stat = [jnp.mean(jnp.abs(s)) for s in stats[key]]

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
