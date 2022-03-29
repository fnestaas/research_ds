"""
This is a script to investigate how to best evaluate the norm of the solution.
To this end, we consider different ways of retrieving the solution values of ODEs
solved by Diffrax. We see that it is considerably faster (.06 vs 15 seconds) to
specify ahead of time which points we would like to evaluate the solution at,
instead of using a dense SaveAt.
"""

from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController
import matplotlib.pyplot as plt 
import jax.numpy as jnp
import time

print('running tests with saveat dense')

vector_field = lambda t, y, args: [-y[1]*10, y[0]*10]
term = ODETerm(vector_field)
solver = Dopri5()
saveat = SaveAt(dense=True)
stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

start = time.time()
sol = diffeqsolve(term, solver, t0=0, t1=3, dt0=0.1, y0=[0, 1], saveat=saveat,
                  stepsize_controller=stepsize_controller)
end = time.time()
print('solving took {} seconds'.format(end - start))

stats = sol.stats
# print(stats)

# print(sol.evaluate(.2))
start = time.time()
vals = [sol.evaluate(i/100) for i in range(100)]
end = time.time()

print('maximum occurs at', jnp.argmax(jnp.array(vals), axis=0))

print('evaluation took {} seconds'.format(end - start))
print('\n=================================\n')
# second approach
print('running tests with saveat discrete')
vector_field = lambda t, y, args: [-y[1]*10, y[0]*10]
term = ODETerm(vector_field)
solver = Dopri5()
saveat = SaveAt(ts=[i/100 for i in range(100)])
stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

start = time.time()
sol = diffeqsolve(term, solver, t0=0, t1=3, dt0=0.1, y0=[0, 1], saveat=saveat,
                  stepsize_controller=stepsize_controller)
end = time.time()
print('solving took {} seconds'.format(end - start))

# print(sol.evaluate(.2))
start = time.time()
vals = jnp.transpose(jnp.array(sol.ys))
end = time.time()

print('maximum occurs at', jnp.argmax(jnp.array(vals), axis=0))

print('evaluation took {} seconds'.format(end - start))


plt.plot(jnp.arange(100), vals)

plt.show()


