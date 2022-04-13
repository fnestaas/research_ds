import jax.numpy as jnp
from jax.experimental.ode import odeint
import flax.linen
from jax import grad, jacrev, jit
import jax.random as random
import matplotlib.pyplot as plt

time = jnp.linspace(0, 1, 100)
xs = jnp.sin(time)


def create_marginally_stable_matrix(n, key, period_bound=jnp.pi / 2):
    a = random.uniform(key=key, shape=(n, n))
    skew = 0.5 * (a - a.T)
    max_eigen_value = jnp.max(jnp.abs(jnp.linalg.eigvals(skew)))
    return skew / max_eigen_value * period_bound


a = create_marginally_stable_matrix(3, random.PRNGKey(0))


def dynamics(x, t):
    # x: state
    # returns: derivative of state
    return a @ x


ddynmaics_dx = jacrev(dynamics)


def adjoint_dynamics(adjoint, x, t):
    return - adjoint @ ddynmaics_dx(x, t)


def diffeq(joint_state, t):
    adjoint = joint_state[:3]
    x = joint_state[3:6]
    return jnp.concatenate(
        [adjoint_dynamics(adjoint, x, t), dynamics(x, t),
         jnp.linalg.norm(adjoint_dynamics(adjoint, x, t)).reshape(1, )])


key = random.PRNGKey(0)
key, subkey = random.split(key)


@jit
def predict(times):
    return odeint(diffeq, jnp.concatenate([random.normal(key=subkey, shape=(6,)), jnp.array(0.0).reshape(1, )]), times)


def loss(x0, times, params, lables):
    # x0: initial state
    # times: array of times at which to predict
    # params: model parameters
    # lables: array of labels
    # returns: loss
    return jnp.mean(jnp.square(predict(x0, times, params) - lables))


if __name__ == '__main__':
    out = predict(time)

    plt.plot(time, out[:, :3], '--', label='adjoint')
    plt.plot(time, out[:, 3:6], label='state')
    plt.plot(time, out[:, -1], label='norm')
    plt.legend()
    plt.show()
