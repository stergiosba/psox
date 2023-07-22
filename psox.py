import chex
from typing import Tuple, Dict, Callable
from jax import numpy as jnp, random as jrandom, jit, lax


@jit
def calculate_best_position(
        f_values: chex.Array,
        best_particle_cost: chex.Array,
        x: chex.Array,
        pb: chex.Array)->  chex.Array:
    """The function calculates the best position for each particle based on the given values and previous
        best positions.

    Args:
        - `f_values (chex.Array)`: Objective function values for each particle.
        - `best_particle_cost (chex.Array)`: Best objective function value for each particle.
        - `x (chex.Array)`: Particle positions.
        - `pb (chex.Array)`: Previous best positions.

    Returns:
        - `new_pb (chex.Array)`: New best positions for each particle.
    """

    bests = f_values<best_particle_cost
    new_pb = bests[:,None]*x + (1-bests[:,None])*pb
    
    return new_pb

@jit
def calculate_velocity(
        key: chex.Array,
        x: chex.Array,
        v: chex.Array,
        pb: chex.Array,
        gb: chex.Array,
        w: float,
        c1: float,
        c2: float)-> chex.Array:
    """Calculates the new velocity for each particle.

    Args:
        - `key (chex.Array)`: A random key.
        - `x (chex.Array)`: Particle positions.
        - `v (chex.Array)`: Particle velocities.
        - `pb (chex.Array)`: Previous best positions.
        - `gb (chex.Array)`: Previous best global position.
        - `w (float)`: Inertia weight.
        - `c1 (float)`: Personal best weight.
        - `c2 (float)`: Global best weight.

    Returns:
        - `new_vel (chex.Array)`: The new velocity for each particle.
    """
    n_dim, d_dim = x.shape
    r1, r2 = jrandom.uniform(key, shape=(2, n_dim, d_dim))
    
    new_vel = w * v + c1 * r1 * (pb - x) + c2 * r2* (gb - x)

    return new_vel

@jit 
def argminx(A: chex.Array)-> Tuple[chex.Array, chex.Array]:
    """Calculates the minimum value and index of an array.

    Args:
        - `A (chex.Array)`: Array to be evaluated.

    Returns:
        - `min, argmin [Tuple(chex.Array, chex.Array)]`: Minimum value and index of the array.
    """
    argmin = jnp.argmin(A)
    return (A[argmin], argmin)

@jit
def calculate_global_best(
        x: chex.Array, 
        best_val: chex.Array, 
        best_global_val: chex.Array, 
        best_index: chex.Array, 
        gb: chex.Array) -> chex.Array:
    """Calculates the best global.

    Args:
        - `x (chex.Array)`: Particle positions.
        - `best_val (chex.Array)`: Best value for this run.
        - `best_global (chex.Array)`: Best global position.
        - `gb (chex.Array)`: Previous best global position.
        - `w (float)`: Inertia weight.
        - `c1 (float)`: Personal best weight.
        - `c2 (float)`: Global best weight.

    Returns:
        - `min, argmin [Tuple(chex.Array, chex.Array)]`: Minimum value and index of the array.
    """
    new_best_global_val = best_val*(best_val < best_global_val) + best_global_val*(best_val >= best_global_val)
    gb = x[best_index]*(best_val < best_global_val) + gb*(best_val >= best_global_val)
    
    return new_best_global_val, gb


def pso(
    key: chex.Array,
    func: Callable[[chex.Array], chex.Array],
    params: Dict,
    scale_x: float=1,
    scale_v: float=1) -> Tuple[chex.Array, chex.Array]:
    
    n_dim = params["n_dim"]
    d_dim = params["d_dim"]
    w = params["w"]
    c1 = params["c1"]
    c2 = params["c2"]

    key, key_v = jrandom.split(key, 2)
    init_X = jrandom.uniform(key, shape=(n_dim, d_dim))*scale_x
    init_V = jrandom.normal(key_v, shape=(n_dim, d_dim))*scale_v
    init_pb = init_X
    fx = func(init_X)
    f_pb = fx
    init_best_gb_val, init_best_gb_index = argminx(fx)
    init_gb = init_X[init_best_gb_index]
    
    @jit
    def body_fun(state, tmp):
        key, X, V, pb, gb, best_global_val = state
        
        f_values = func(X)
        run_best_value, run_best_index = argminx(f_values)
                
        pb = calculate_best_position(f_values, f_pb, X, pb)
        best_global, gb = calculate_global_best(X, run_best_value, best_global_val, run_best_index, gb)

        V = calculate_velocity(key, X, V, pb, gb, w, c1, c2)
        key, _ = jrandom.split(key, 2)
        pb = X
            
        carry = [key, X+V, V, pb, gb, best_global]
        y = [gb,best_global_val]
        return carry, y
    
    _, scan_out = lax.scan(body_fun, [key, init_X, init_V, init_pb, init_gb, init_best_gb_val], (), params["timesteps"])
    
    gb = scan_out[0]
    best_global = scan_out[1]
    
    return gb[-1], best_global[-1]
