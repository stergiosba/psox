from jax import numpy as jnp, random as jrandom, jit

@jit
def calculate_best_position(f_values, best_particle_cost, x, pb):
    bests = f_values<best_particle_cost
    pos = bests[:,None]*x + (1-bests[:,None])*pb
    return pos

@jit
def calculate_velocity(key, x, v, pb, gb):
    n_dim, d_dim = x.shape
    w = 0.9
    c1 = 0.5
    c2 = 0.3
    r1, r2 = jrandom.uniform(key, shape=(2, n_dim, d_dim))
    inertia = w * v
    best_particle_pos_component = r1 * (pb - x)
    best_global_pos_component = r2 * (gb - x)

    new_velocity = inertia + c1 * best_particle_pos_component + c2 * best_global_pos_component
    return new_velocity

@jit 
def argminx(x):
    argmin = jnp.argmin(x)
    return (x[argmin], argmin)

def run(key, func, n_dim, d_dim, timesteps, scale_x=1, scale_v=1):
    
    key, key_v = jrandom.split(key, 2)
    X = jrandom.uniform(key, shape=(n_dim, d_dim))*scale_x
    X = jnp.clip(X, 7, 50)
    V = jrandom.normal(key_v, shape=(n_dim, d_dim))*scale_v
    pb = X
    f_pb = func(pb)
    best_global, gb_index = argminx(func(X))
    gb = X[gb_index]
    
    for k in range(timesteps):
        f_values = func(X)
        best_value, best_index = argminx(f_values)
        
        # particles x dimensions
        pb = calculate_best_position(f_values, f_pb, X, pb)
        
        #gb = X[best_index]*(best_value < best_global) + gb*(best_value >= best_global)
        
        if best_value < best_global:
            # Update best swarm cost and position
            best_global = best_value
            gb = X[best_index]
        
        key, _ = jrandom.split(key, 2)

        V = calculate_velocity(key, X, V, pb, gb)

        X+=V
        X = jnp.clip(X, -7, 50)
        pb = X
        
        if k%100==0:
            print(f"Step: {k}, Best value: {best_global}, Best position: {gb}")

    return gb

def main():
    @jit
    def expx2x1(x):
        y = -jnp.exp(-(x**2+x+1))
        return y.flatten()
    
    n_dim = 300
    d_dim = 1
    timesteps = 500
    key = jrandom.PRNGKey(0)

    run(key, expx2x1, n_dim, d_dim, timesteps)
        
    
if __name__ == "__main__":
    main()