# PSOX

**PSOX** is an efficient implementation of the vanilla Particle Swarm Optimization solver on Jax.

## How to use:

- First define a cost function that is to be optimized. The function needs to be scalar!
- Select parameters for the swarm
- Select timesteps to run the algorithm for

An example looks like the following:

```python
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

```

The function to be minimized can be jitted for faster calculations and for accuracy purposes. 