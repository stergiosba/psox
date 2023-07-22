from jax import numpy as jnp, random as jrandom, jit
from psox import pso

def main():
    @jit
    def func(x):
        y = -jnp.exp(-((x)**2))
        return jnp.sum(y, axis=1)

    params = {
        "n_dim": 300,
        "d_dim": 2,
        "timesteps": 150+15*5,
        "w": 0.75,
        "c1": 0.1,
        "c2": 0.1
    }

    key = jrandom.PRNGKey(0)

    res = pso(key, func, params)

    print(f"Best value: {res[1]}, Best position: {res[0]}")
    
if __name__ == "__main__":
    main()