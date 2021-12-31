
from jax.config import config
config.update("jax_enable_x64", True)

import context

import numpy as np 
import jax 
import jax.numpy as jnp

import jax_hps.cheb as cheb

import matplotlib.pyplot as plt

# create source
def source(x,y):
    return -2*(jnp.pi)**2*jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y)

def sol(x,y):
    return jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y)

# create the grid and operator
nx = 20
ny = 20

(lap, x_grid, y_grid, idx_int, idx_bdy) = cheb.cheb_2d_lap(nx,ny)

lap_int_bdy = lap[idx_int,:][:,idx_bdy] 

u_bdy = sol(x_grid.reshape((-1,1))[idx_bdy],\
            y_grid.reshape((-1,1))[idx_bdy]).reshape((-1,1))

# preparing the source (we evaluate it everywhere)
f_int = source(x_grid, y_grid).reshape((-1,1))[idx_int]

s_int = f_int - lap_int_bdy @ u_bdy

lap_int = lap[idx_int,:][:,idx_int]

X_tau = jnp.linalg.inv(lap_int)

u_int = X_tau @ f_int

u = jnp.zeros((ny+1, nx+1)).reshape((-1,1))

u = u.at[idx_int].set(u_int)
u = u.at[idx_bdy].set(u_bdy)

u = u.reshape((ny+1, nx+1))
u_ref = sol(x_grid, y_grid)

print("Error with respect to the solution: %e"% jnp.linalg.norm(u - u_ref))

plt.imshow(jnp.abs(u - u_ref))
plt.colorbar()
plt.title("error")

plt.show()