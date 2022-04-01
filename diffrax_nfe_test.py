import jax.numpy as jnp 
import joblib 
import matplotlib.pyplot as plt 

stats = joblib.load('stats.pkl')
key = 'num_steps' # 'state_norm' 
N = len(stats[key])

stat = [jnp.mean(s) for s in stats[key]]

fig, ax = plt.subplots(1, 1)

ax.plot(
    jnp.arange(N), 
    stat,
    )

ax.set_xlabel('Epoch')
ax.set_ylabel(key)

plt.show()
