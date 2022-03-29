import jax.numpy as jnp 
import joblib 
import matplotlib.pyplot as plt 

nfes = joblib.load('nfe.pkl')
N = len(nfes)

mean_nfe = [jnp.mean(n) for n in nfes]

fig, ax = plt.subplots(1, 1)

ax.plot(
    jnp.arange(N), 
    mean_nfe,
    )

ax.set_xlabel('Epoch')
ax.set_ylabel('Mean NFE')

plt.show()
