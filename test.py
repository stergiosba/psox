from jax import numpy as jnp, random as jrandom, jit
from psox import run

def main():
    @jit
    def expx2x1(x):
        y = -jnp.exp(-(x**2+x+1))
        return y.flatten()

    n_dim = 300
    d_dim = 1
    timesteps = 500
    w = 0.9
    c1 = 0.5
    c2 = 0.3
    key = jrandom.PRNGKey(0)

    run(key, expx2x1, n_dim, d_dim, w, c1, c2, timesteps)
    
    
if __name__ == "__main__":
    main()