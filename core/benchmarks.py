import jax.numpy as jnp
import numpy as np
from jax import jit

from .jax_gd_pso import jax_gd_pso
from .pso import pso


def schwefel_np(x: np.ndarray) -> float:
    n = x.shape[0]
    sum_term = np.sum(x * np.sin(np.sqrt(np.abs(x))))
    return 418.9829 * n - sum_term


def rastrigin_np(x: np.ndarray) -> float:
    n = x.shape[0]
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def sphere_np(x: np.ndarray) -> float:
    return np.sum(x**2)


def elliptic_np(x: np.ndarray) -> float:
    n = x.shape[0]
    i = np.arange(n)
    coeffs = (1e6) ** (i / (n - 1))
    return np.sum(coeffs * x**2)


@jit
def schwefel_jax(x: jnp.ndarray) -> jnp.ndarray:
    n = x.shape[0]
    sum_term = jnp.sum(x * jnp.sin(jnp.sqrt(jnp.abs(x))))
    return 418.9829 * n - sum_term


@jit
def rastrigin_jax(x: jnp.ndarray) -> jnp.ndarray:
    n = x.shape[0]
    return 10 * n + jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x))


@jit
def sphere_jax(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(x**2)


@jit
def elliptic_jax(x: jnp.ndarray) -> jnp.ndarray:
    n = x.shape[0]
    i = jnp.arange(n)

    coeffs = (1e6) ** (i / (n - 1))
    return jnp.sum(coeffs * x**2)


BENCHMARKS = {
    "Schwefel": {
        "bounds": (-500, 500),
        "PSO": schwefel_np,
        "JAX-GD-PSO": schwefel_jax,
    },
    "Rastrigin": {
        "bounds": (-5.12, 5.12),
        "PSO": rastrigin_np,
        "JAX-GD-PSO": rastrigin_jax,
    },
    "Elliptic": {
        "bounds": (-5.0, 10.0),
        "PSO": elliptic_np,
        "JAX-GD-PSO": elliptic_jax,
    },
    "Sphere": {
        "bounds": (-5.12, 5.12),
        "PSO": sphere_np,
        "JAX-GD-PSO": sphere_jax,
    },
}

ALGORITHMS = {
    "PSO": pso,
    "JAX-GD-PSO": jax_gd_pso,
}

DIMS = [30, 100, 500, 1000]

HYPERPARAMETERS = {
    "num_dims": None,
    "num_particles": 30,
    "max_iters": 1000,
    "c1": 1.5,
    "c2": 2.5,
    "w": 0.7,
    "seed": None,
    "eta": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "weight_decay": 0.01,
    "steps": 10,
}

NUM_RUNS = 10
