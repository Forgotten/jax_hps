
import jax 
from jax.config import config
config.update("jax_enable_x64", True)

import context

import numpy as np 
import jax 
import jax.numpy as jnp

import jax_hps.cheb as cheb

import matplotlib.pyplot as plt

import time

offset = 0.5
# create source
def source(x,y):
    return -2*(jnp.pi)**2*jnp.sin(jnp.pi*(x + offset))*jnp.sin(jnp.pi*y)

def sol(x,y):
    return jnp.sin(jnp.pi*(x + offset))*jnp.sin(jnp.pi*y)

def dx_sol(x,y):
    return jnp.pi*jnp.cos(jnp.pi*(x + offset))*jnp.sin(jnp.pi*y)

def dy_sol(x,y):
    return jnp.pi*jnp.sin(jnp.pi*(x + offset))*jnp.cos(jnp.pi*y)

# create the grid and operator
nx = 20
ny = 20


# defining the leaves

x_coord, y_coord = jnp.array([-7., -5., -3., -1., 1., 3., 5., 7.]),\
                   jnp.array([-7., -5., -3., -1., 1., 3., 5., 7.])

box_lvl_0 = [[ cheb.init_leaf_jax(nx,ny, [x, y]) for x in x_coord] for y in y_coord]

####################################################################
#                            Merging
####################################################################

# one level of merges
box_lvl_1 = []
for row in box_lvl_0:
    box_lvl_1.append([cheb.init_box_jax(c, c1) for c, c1 in zip(row[::2], row[1::2])])

box_lvl_2 = []
for col1, col2 in zip(box_lvl_1[::2], box_lvl_1[1::2]):
    box_lvl_2.append([cheb.init_box_jax(r1, r2) for r1, r2 in zip(col1, col2)])

# the second level of merging
box_lvl_3 = []
for row in box_lvl_2:
    box_lvl_3.append([cheb.init_box_jax(c, c1) for c, c1 in zip(row[::2], row[1::2])])

box_lvl_4 = []
for col1, col2 in zip(box_lvl_3[::2], box_lvl_3[1::2]):
    box_lvl_4.append([cheb.init_box_jax(r1, r2) for r1, r2 in zip(col1, col2)])

# the third level of merging
box_lvl_5 = []
for row in box_lvl_4:
    box_lvl_5.append([cheb.init_box_jax(c, c1) for c, c1 in zip(row[::2], row[1::2])])

box_lvl_6 = []
for col1, col2 in zip(box_lvl_5[::2], box_lvl_5[1::2]):
    box_lvl_6.append([cheb.init_box_jax(r1, r2) for r1, r2 in zip(col1, col2)])


# now we do the computationally heavy part of merging the operators
# one level of merges


update_box_one_level_jit = jax.jit(cheb.update_box_one_level)

tic = time.perf_counter()

(box_lvl_1, box_lvl_2) = update_box_one_level_jit(box_lvl_0, box_lvl_1, box_lvl_2)
(box_lvl_3, box_lvl_4) = update_box_one_level_jit(box_lvl_2, box_lvl_3, box_lvl_4)
(box_lvl_5, box_lvl_6) = update_box_one_level_jit(box_lvl_4, box_lvl_5, box_lvl_6)

toc = time.perf_counter()

print("updating the boxes took %e [s]"% (toc-tic))


####################################################################
#                         Upward pass
####################################################################

# create the loads for leaves
rhs_grid = [[ source(leaf.x_grid, leaf.y_grid).reshape((-1,))\
              for leaf in leaves] for leaves in box_lvl_0]

m_lvl_0 = len(box_lvl_0)
n_lvl_0 = len(box_lvl_0[0])

# jitting the functions 
solve_leaves_jax_jit = jax.jit(cheb.solve_leaves_jax)
solve_up_one_level_jax_jit = jax.jit(cheb.solve_up_one_level_jax)

w_int_lvl_0, h_bdy_lvl_0 = solve_leaves_jax_jit(box_lvl_0, rhs_grid)

# first and second levels

(w_int_lvl_1, w_int_lvl_2),\
(h_bdy_lvl_1, h_bdy_lvl_2) = solve_up_one_level_jax_jit(box_lvl_1, box_lvl_2, h_bdy_lvl_0)

# third and fourth levels

(w_int_lvl_3, w_int_lvl_4),\
(h_bdy_lvl_3, h_bdy_lvl_4) = solve_up_one_level_jax_jit(box_lvl_3, box_lvl_4, h_bdy_lvl_2)

# fifth and sixth levels
(w_int_lvl_5, w_int_lvl_6),\
(h_bdy_lvl_5, h_bdy_lvl_6) = solve_up_one_level_jax_jit(box_lvl_5, box_lvl_6, h_bdy_lvl_4)

####################################################################
#                         Downward pass
####################################################################

# sampling the dirichlet boundary condition
f_bdy = sol(box_lvl_6[0][0].x_grid, box_lvl_6[0][0].y_grid)

m_lvl_6, n_lvl_6 = len(box_lvl_6), len(box_lvl_6[0])

u_bdy_lvl_6 = jnp.zeros((m_lvl_6, n_lvl_6, box_lvl_6[0][0].H_tau.shape[0]))
u_bdy_lvl_6 = u_bdy_lvl_6.at[0,0,:].set(f_bdy)

# jitting the functions 
solve_int_jax_jit = jax.jit(cheb.solve_int_jax)
solve_down_one_level_jax_jit = jax.jit(cheb.solve_down_one_level_jax)

# fifth and sixth levels
u_bdy_lvl_4 = solve_down_one_level_jax_jit(box_lvl_4, box_lvl_5, box_lvl_6,
                                           w_int_lvl_5, w_int_lvl_6, u_bdy_lvl_6)

# third and fourth levels
u_bdy_lvl_2 = solve_down_one_level_jax_jit(box_lvl_2, box_lvl_3, box_lvl_4,
                                        w_int_lvl_3, w_int_lvl_4, u_bdy_lvl_4)

# first and second levels
u_bdy_lvl_0 = solve_down_one_level_jax_jit(box_lvl_0, box_lvl_1, box_lvl_2,
                                        w_int_lvl_1, w_int_lvl_2, u_bdy_lvl_2)

# solving the interior 
u_int = solve_int_jax_jit(box_lvl_0, w_int_lvl_0, u_bdy_lvl_0)


####################################################################
#                    Comparing the Solutions
####################################################################


# reference solution 
# TODO: we need to define a function for this
sol_int = jnp.zeros((m_lvl_0, n_lvl_0, (nx-1)**2))
for i in range(m_lvl_0):
    for j in range(n_lvl_0):
        sol_i =  sol(box_lvl_0[i][j].x_grid, box_lvl_0[i][j].y_grid).reshape((-1,))
        sol_int = sol_int.at[i,j,:].set(sol_i[box_lvl_0[i][j].idx_int])

err = sol_int - u_int

print("Error of the solution is %e"% jnp.linalg.norm(err))

# error in one panel

plt.figure(10)
plt.imshow(sol_int[1,1,:].reshape(((nx-1),(nx-1))))
plt.colorbar()

plt.figure(11)
plt.imshow(u_int[1,1,:].reshape(((nx-1),(nx-1))))
plt.colorbar()


plt.figure(12)

plt.scatter(box_lvl_3[0][0].x_grid, 
            box_lvl_3[0][0].y_grid)

plt.scatter(box_lvl_2[0][0].x_grid[box_lvl_3[0][0].idx_a_3], 
            box_lvl_2[0][0].y_grid[box_lvl_3[0][0].idx_a_3])
            
plt.scatter(box_lvl_2[0][1].x_grid[box_lvl_3[0][0].idx_b_3], 
            box_lvl_2[0][1].y_grid[box_lvl_3[0][0].idx_b_3])

# plt.figure(12)

plt.scatter(box_lvl_1[1][0].x_grid[box_lvl_2[0][0].idx_b_3], 
            box_lvl_1[1][0].y_grid[box_lvl_2[0][0].idx_b_3])

plt.scatter(box_lvl_1[0][0].x_grid[box_lvl_2[0][0].idx_a_3], 
            box_lvl_1[0][0].y_grid[box_lvl_2[0][0].idx_a_3])

plt.scatter(box_lvl_1[0][1].x_grid[box_lvl_2[0][1].idx_a_3], 
            box_lvl_1[0][1].y_grid[box_lvl_2[0][1].idx_a_3])

plt.scatter(box_lvl_1[1][1].x_grid[box_lvl_2[0][1].idx_b_3], 
            box_lvl_1[1][1].y_grid[box_lvl_2[0][1].idx_b_3])



plt.show()