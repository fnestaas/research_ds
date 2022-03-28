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
N = len(nfes)//2

mean_nfe = [jnp.mean(n) for n in nfes[::2]]

plt.plot(np.arange(N), mean_nfe)
plt.show()



# vector_field = lambda t, y, args: -y * jnp.exp(y)

# term = ODETerm(vector_field) # -y dt term
# solver = Dopri5()
# saveat = SaveAt(
#         ts=[0, .25, .5, .7, 1], 
#         # solver_state=True, 
#     )
# stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

# sol = diffeqsolve(
#         term, 
#         solver, 
#         t0=0, 
#         t1=1, 
#         dt0=.1, 
#         y0=1, 
#         saveat=saveat, 
#         stepsize_controller=stepsize_controller, 
#     )

# stats = sol.stats

# print('\n\nNum steps =', stats['num_steps'])
