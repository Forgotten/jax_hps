
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

x_coord, y_coord = jnp.array([-3., -1., 1., 3.]), jnp.array([-3., -1., 1., 3.])

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

# now we do the computationally heavy part of merging the operators
# one level of merges


update_box_one_level_jit = jax.jit(cheb.update_box_one_level)

tic = time.perf_counter()

(box_lvl_1, box_lvl_2) = update_box_one_level_jit(box_lvl_0, box_lvl_1, box_lvl_2)
(box_lvl_3, box_lvl_4) = update_box_one_level_jit(box_lvl_2, box_lvl_3, box_lvl_4)

toc = time.perf_counter()

print("updating the boxes took %e [s]"% (toc-tic))

# update_box_jax_jit = jax.jit(cheb.update_box_jax)

# for i in range(len(box_lvl_1)):
#     for j in range(len(box_lvl_1[0])):
#         box_lvl_1[i][j] = update_box_jax_jit(box_lvl_1[i][j], 
#                                              box_lvl_0[i][2*j],
#                                              box_lvl_0[i][2*j+1])

# for i in range(len(box_lvl_2)):
#     for j in range(len(box_lvl_2[0])):
#         box_lvl_2[i][j] = update_box_jax_jit(box_lvl_2[i][j], 
#                                              box_lvl_1[2*i][j],
#                                              box_lvl_1[2*i+1][j])


# (box_lvl_1, box_lvl_2) = update_box_one_level_jit(box_lvl_0, box_lvl_1, box_lvl_2)


# for i in range(len(box_lvl_3)):
#     for j in range(len(box_lvl_3[0])):
#         box_lvl_3[i][j] = update_box_jax_jit(box_lvl_3[i][j], 
#                                              box_lvl_2[i][2*j],
#                                              box_lvl_2[i][2*j+1])

# for i in range(len(box_lvl_4)):
#     for j in range(len(box_lvl_4[0])):
#         box_lvl_4[i][j] = update_box_jax_jit(box_lvl_4[i][j], 
#                                              box_lvl_3[2*i][j],
#                                              box_lvl_3[2*i+1][j])

##################################################################
# ploting the differnt meshes

plt.figure(1)
plt.title("grid in the panels")
for row in box_lvl_0:
    for r in row:
        plt.scatter(r.x_grid, r.y_grid)

plt.figure(2)
plt.title("grid in the boundary of the first level")
for row in box_lvl_1:
    for r in row:
        plt.scatter(r.x_grid, r.y_grid)

plt.figure(3)
plt.title("grid in the boundary of the second level")
for row in box_lvl_2:
    for r in row:
        plt.scatter(r.x_grid, r.y_grid)

plt.figure(4)
plt.title("grid in the boundary of the third level")
for row in box_lvl_3:
    for r in row:
        plt.scatter(r.x_grid, r.y_grid)

plt.figure(5)
plt.title("grid in the boundary of the root level")
plt.scatter(box_lvl_4[0][0].x_grid, box_lvl_4[0][0].y_grid)

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

####################################################################
#                         Downward pass
####################################################################

# sampling the dirichlet boundary condition
f_bdy = sol(box_lvl_4[0][0].x_grid, box_lvl_4[0][0].y_grid)

m_lvl_4, n_lvl_4 = len(box_lvl_4), len(box_lvl_4[0])

u_bdy_lvl_4 = jnp.zeros((m_lvl_4, n_lvl_4, box_lvl_4[0][0].H_tau.shape[0]))
u_bdy_lvl_4 = u_bdy_lvl_4.at[0,0,:].set(f_bdy)

# jitting the functions 
solve_int_jax_jit = jax.jit(cheb.solve_int_jax)
solve_down_one_level_jax_jit = jax.jit(cheb.solve_down_one_level_jax)

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
sol_int = jnp.zeros((4,4,(nx-1)**2))
for i in range(4):
    for j in range(4):
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