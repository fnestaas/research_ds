"""
Goal: experiment with Diffrax's adjoint methods, described under
https://docs.kidger.site/diffrax/api/adjoints/
to answer the questions of W02:
Plot grad wrt theta(t) and gradient of state (i.e. d(x(T))/d(x(0)), or dL/dz(0). see diffrax adjoint sensitivity)
"""

from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController
import diffrax
import matplotlib.pyplot as plt 
import jax.numpy as jnp
import time
import equinox as eqx
import jax

class ODE_Sol(eqx.Module):
    y0: jnp.array 

    def __init__(self, y0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.y0 = y0 
    
    def __call__(self, t):
        vector_field = lambda t, y, args: jnp.array([-y[1]*10, y[0]*10])
        term = ODETerm(vector_field)
        solver = Dopri5()
        saveat = SaveAt(ts=[0, t])
        stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

        y0 = self.y0

        sol_rca = diffeqsolve(
            term, 
            solver, 
            t0=0, 
            t1=t, 
            dt0=0.1, 
            y0=y0, 
            saveat=saveat, 
            stepsize_controller=stepsize_controller, 
            adjoint=diffrax.RecursiveCheckpointAdjoint()
            )
        
        return sol_rca.ys[-1, :] # return solution at final time

@eqx.filter_value_and_grad
def grad_wrt_init(model, t):
    """
    return the final value of dimension -1
    """
    y = model(t)
    return y[0]

def f(y0):
    t = 1
    vector_field = lambda t, y, args: jnp.array([-y[1]*10, y[0]*10])
    term = ODETerm(vector_field)
    solver = Dopri5()
    saveat = SaveAt(ts=[0, t])
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

    sol_rca = diffeqsolve(
        term, 
        solver, 
        t0=0, 
        t1=t, 
        dt0=0.1, 
        y0=y0, 
        saveat=saveat, 
        stepsize_controller=stepsize_controller, 
        adjoint=diffrax.RecursiveCheckpointAdjoint()
        )
    
    return sol_rca.ys[-1, :] # return solution at final time

y0 = jnp.array([0., 1.])
ode_sol = ODE_Sol(y0=y0)
# f, g = grad_wrt_init(ode_sol, t=1.)
# grad = jax.jacrev(ode_sol, 0)(y0)
grad = jax.jacrev(f)
result = grad(y0)


"""
The following is all that I tried out in order to solve the problem
"""

# def to_diff(y0):
#     vector_field = lambda t, y, args: jnp.array([-y[1]*10, y[0]*10])
#     term = ODETerm(vector_field)
#     solver = Dopri5()
#     saveat = SaveAt(dense=True)
#     stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

#     y0 = jnp.array([0, 1])

#     sol_rca = diffeqsolve(
#         term, 
#         solver, 
#         t0=0, 
#         t1=1, 
#         dt0=0.1, 
#         y0=y0, 
#         saveat=saveat, 
#         stepsize_controller=stepsize_controller, 
#         adjoint=diffrax.RecursiveCheckpointAdjoint()
#         )
#     return sol_rca.evaluate

# y, g_rca = jax.vjp(to_diff, 0)



# vector_field = lambda t, y, args: jnp.array([-y[1]*20 + jnp.exp(y[0]) + jnp.exp(y[1]), y[0]*100])
# term = ODETerm(vector_field)
# solver = Dopri5()
# saveat = SaveAt([0, 1])
# stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

# y0 = jnp.array([0, 1])

# sol_rca = diffeqsolve(
#     term, 
#     solver, 
#     t0=0, 
#     t1=1, 
#     dt0=0.1, 
#     y0=y0, 
#     saveat=saveat, 
#     stepsize_controller=stepsize_controller, 
#     adjoint=diffrax.RecursiveCheckpointAdjoint()
#     )

# _, g_rca = grad_wrt_init(sol_rca.ys)

# sol_cont = diffeqsolve(
#     term, 
#     solver, 
#     t0=0, 
#     t1=1, 
#     dt0=0.1, 
#     y0=jnp.array([0, 1]), 
#     saveat=saveat, 
#     stepsize_controller=stepsize_controller, 
#     adjoint=diffrax.BacksolveAdjoint(solver=Dopri5()), 
#     )

a=0
