# In this file we try to find how many function evaluations diffrax needs to solve an ODE

import jax.numpy as jnp 
import jax.random as jrandom 
import jax 
from diffrax import ODETerm, PIDController, diffeqsolve, Dopri5, SaveAt
from numpy import save
import joblib 
import matplotlib.pyplot as plt 
import numpy as np

nfes = joblib.load('nfe.pkl') # really strange structure
N = len(nfes)

mean_nfe = [jnp.mean(n) for n in nfes]

fig, ax = plt.subplots(1, 1)

ax.plot(
    np.arange(N), 
    mean_nfe,
    )

ax.set_xlabel('Epoch')
ax.set_ylabel('Mean NFE')

plt.show()