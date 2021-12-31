# script to generate the chebyshev differentiation matrices

from re import U
import jax
from jax._src.lax.lax import Array 
from jax.config import config
config.update("jax_enable_x64", True)

from jax_hps import dataclasses

from jax import jit
import jax.numpy as jnp

# importing the typing
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, Union

#TODO: we need to rescale for different lenghts of domain
def cheb_1d_diff(N: jnp.int64, lenght:jnp.float64 = 2.0):
    # from Trefethen's book

    # if N is greater than zero
    if N : 

        x = jnp.cos(jnp.linspace(0, jnp.pi, N+1)).reshape((-1,1))
        c = (jnp.hstack(([[2.], jnp.ones((N-1)), [2.]]))\
             *(jnp.array([-1])**(jnp.linspace(0, N, N+1, 
                                 dtype=jnp.int64)))).reshape((-1, 1))

        X = x*jnp.ones((1, N+1))

        dX = X - X.T

        D = c.dot(1/c.T)/(dX + jnp.eye(N+1))

        D = D - jnp.diag(jnp.sum(D.T, axis = 0), 0)

        ratio = lenght/2.0

        return (D/ratio, x*ratio)

    else: 
        return (0, 1)


def cheb_2d_lap(nx:jnp.int64, ny:jnp.int64, 
                lenght_x:jnp.float64 = 2.0,
                lenght_y:jnp.float64 = 2.0):

    # computes the laplacian using chebychev grids
    (dudx, x_grid) =  cheb_1d_diff(nx, lenght_x)
    (dudy, y_grid) =  cheb_1d_diff(ny, lenght_y)
    
    ddudxx = dudx.dot(dudx)
    ddudyy = dudy.dot(dudy)

    lap = jnp.kron(jnp.eye(ny+1), ddudxx) + jnp.kron(ddudyy, jnp.eye(nx+1))

    x_grid_2d = jnp.ones((ny+1, 1))*x_grid.T
    y_grid_2d = jnp.ones((1, nx+1))*y_grid

    # now we extract the boundary indices 
    # todo: extract them and sort them in North, South, East and West
    idx = jnp.linspace(0, (nx+1)*(ny+1)-1, (nx+1)*(ny+1),
                       dtype=jnp.int64).reshape((ny+1,nx+1))

    ratio = lenght_x/2.0
    tol = 1e-4*ratio

    bdy_i, bdy_j = jnp.where( jnp.isclose(x_grid_2d, 1.0*ratio, rtol=tol)\
                            + jnp.isclose(x_grid_2d,-1.0*ratio, rtol=tol)\
                            + jnp.isclose(y_grid_2d, 1.0*ratio, rtol=tol)\
                            + jnp.isclose(y_grid_2d,-1.0*ratio, rtol=tol))

    idx_bdy = idx[bdy_i,bdy_j].sort()
    idx_int = jnp.setdiff1d(idx.reshape((-1,)), idx_bdy, assume_unique=True)

    return (lap, x_grid_2d, y_grid_2d, idx_int, idx_bdy) 



def cheb_2d_dtn(nx, ny):
    # computes the laplacian using chebychev grids
    (dudx, x_grid) =  cheb_1d_diff(nx)
    (dudy, y_grid) =  cheb_1d_diff(ny)
    
    ddudxx = dudx.dot(dudx)
    ddudyy = dudy.dot(dudy)

    # first order derivatives in the domain
    dx = jnp.kron(jnp.eye(ny+1), dudx)
    dy = jnp.kron(dudy, jnp.eye(nx+1))

    # laplacian
    lap = jnp.kron(jnp.eye(ny+1), ddudxx) + jnp.kron(ddudyy, jnp.eye(nx+1))

    x_grid_2d = jnp.ones((ny+1, 1))*x_grid.T
    y_grid_2d = jnp.ones((1, nx+1))*y_grid

    # now we extract the boundary indices 
    # extract them and sort them in North, South, East and West
    # we also remove the corner points
    idx = jnp.linspace(0, (nx+1)*(ny+1)-1, (nx+1)*(ny+1),
                       dtype=jnp.int64).reshape((ny+1,nx+1))

    # East
    bdy_i, bdy_j = jnp.where(jnp.isclose(x_grid_2d[1:-1,:], 1.0, rtol = 1e-4))
    idx_bdy_e = idx[1:-1,:][bdy_i,bdy_j].sort()

    # West
    bdy_i, bdy_j = jnp.where(jnp.isclose(x_grid_2d[1:-1,:],-1.0, rtol = 1e-4))
    idx_bdy_w = idx[1:-1,:][bdy_i,bdy_j].sort()

    # North
    bdy_i, bdy_j = jnp.where(jnp.isclose(y_grid_2d[:,1:-1], 1.0, rtol = 1e-4))
    idx_bdy_n = idx[:,1:-1][bdy_i,bdy_j].sort()

    # South
    bdy_i, bdy_j = jnp.where(jnp.isclose(y_grid_2d[:,1:-1],-1.0, rtol = 1e-4))
    idx_bdy_s = idx[:,1:-1][bdy_i,bdy_j].sort()

    # TODO: restrict the boundary terms

    idx_bdy = jnp.hstack([idx_bdy_s, 
                          idx_bdy_e, 
                          idx_bdy_n, 
                          idx_bdy_w ])

    # easier way is to compute the boundary terms, and then use setdiff to 
    # remove the indices corresponsing to the corner points

    idx_corners = jnp.array([idx[0,0], idx[0, -1], idx[-1,0], idx[-1,-1]])
    idx_int = jnp.setdiff1d(idx.reshape((-1,)), idx_bdy, assume_unique = True)

    # we remove the corners
    idx_int = jnp.setdiff1d(idx_int, idx_corners, assume_unique = True)

    return (lap, dx, dy, x_grid_2d, y_grid_2d, idx_int,
            (idx_bdy, idx_bdy_s, idx_bdy_e, idx_bdy_n, idx_bdy_w))


def cheb_2d_poincare_steklov(nx, ny):
    # computes the laplacian using chebychev grids
    
    # building the matrices
    (lap, dx, dy,
    x_grid, y_grid,
    idx_int, idx_bdy) = cheb_2d_dtn(nx, ny)

    # building the operators

    lap_int = lap[idx_int,:][:,idx_int]

    X_tau = jnp.linalg.inv(lap_int)

    # we create the degrees of freedom of the interior
    U = jnp.zeros((lap.shape[0], len(idx_int)))

    # we only set the ones in the boundary
    U = U.at[idx_int].set(X_tau)

    # neumann matrix 
    N = jnp.vstack([ dy[idx_bdy[1], :],
                     dx[idx_bdy[2], :],
                     dy[idx_bdy[3], :],
                     dx[idx_bdy[4], :]])

    # computing the load to neumann map
    H_tau = N @ U

    # here we compute the Dtn map

    # define A_{i,b}
    lap_int_bdy = lap[idx_int,:][:,idx_bdy[0]] 

    # build the source term 
    phi = jnp.eye(idx_bdy[0].shape[0])
    S_int = - lap_int_bdy @ phi
    U_int = X_tau @ S_int

    U = jnp.zeros(((ny+1)*(nx+1), idx_bdy[0].shape[0]))

    U = U.at[idx_int].set(U_int)
    U = U.at[idx_bdy[0]].set(phi)

    T_tau = N @ U 

    # we keep track of the normal pointing outside
    # and outside
    # Theta_tau = jnp.ones((len(idx_bdy[0]),), dtype = jnp.int64)
    # Theta_tau = Theta_tau.at[idx_bdy[1]].set(-1)
    # Theta_tau = Theta_tau.at[idx_bdy[4]].set(-1)

    operators = {"T_tau": T_tau,
                 "H_tau": H_tau,
                 "X_tau": X_tau,
                 "S_tau": U_int}
                #  "Theta_tau": Theta_tau}

    return ( operators,  # operators 
            (lap, dx, dy),          # differenciation
            (x_grid, y_grid), # grids
            idx_int,                # interior indices
            idx_bdy)

class leaf():
    def __init__(self, nx, ny, center):

        (operators, Diff,
        grid,
        idx_int, idx_bdy) = cheb_2d_poincare_steklov(nx, ny)

        self.T_tau = operators["T_tau"]
        self.H_tau = operators["H_tau"]
        self.X_tau = operators["X_tau"]
        self.S_tau = operators["S_tau"]
        # self.Theta_tau = operators["Theta_tau"]

        self.x_grid = grid[0] + center[0]
        self.y_grid = grid[1] + center[1]

        self.idx_int = idx_int
        self.idx_bdy = idx_bdy

    def solve(self, u_bdy):
        return self.S_tau @ u_bdy

    def solve_load_up(self, g):
        # computing for the load
        g_int = g[self.idx_int]
        # compute derivatives of the local solution
        h_bdy = self.H_tau @ g_int
        # compute the interior of the load problem
        w_int = self.X_tau @ g_int

        return (w_int, h_bdy)

    def solve_load_down(self, w_int, u_bdy):
        return self.solve(u_bdy) + w_int
    

class box():
    def __init__(self, child_alpha, child_beta):
        # compute the indices in common
        idx_bdy_alpha = child_alpha.idx_bdy[0]
        idx_bdy_beta  = child_beta.idx_bdy[0]

        # extract the grids, we reshape it so they are in the 
        # correct format, this is for the leafs for which the 
        # grids fields contain the dimensions of the rectangle
        x_grid_alpha = child_alpha.x_grid.reshape((-1,))
        y_grid_alpha = child_alpha.y_grid.reshape((-1,))
        
        x_grid_beta = child_beta.x_grid.reshape((-1,))
        y_grid_beta = child_beta.y_grid.reshape((-1,))

        diff_x = x_grid_alpha[idx_bdy_alpha].reshape((-1,1))\
               - x_grid_beta[idx_bdy_beta].reshape((1,-1)) 
        
        diff_y = y_grid_alpha[idx_bdy_alpha].reshape((-1,1))\
               - y_grid_beta[idx_bdy_beta].reshape((1,-1)) 

        diff_norm = jnp.sqrt(jnp.square(diff_x) + jnp.square(diff_y))

        # we want the indices in which the two values are the same
        idx_a_3, idx_b_3 = jnp.where(jnp.isclose(diff_norm, 0.))

        # we make sure that the indices are correct, i.e. that they correspond
        # to the same grid points
        assert jnp.isclose(jnp.linalg.norm(x_grid_alpha[idx_bdy_alpha][idx_a_3]\
                                          - x_grid_beta[idx_bdy_beta][idx_b_3]),
                            0.0)
        assert jnp.isclose(jnp.linalg.norm(y_grid_alpha[idx_bdy_alpha][idx_a_3]\
                                          - y_grid_beta[idx_bdy_beta][idx_b_3]),
                            0.0)

        # split the different indices
        idx_a_1 = jnp.linspace(0, len(idx_bdy_alpha)-1, len(idx_bdy_alpha),
                               dtype=jnp.int64)
        idx_b_2 = jnp.linspace(0, len(idx_bdy_beta)-1, len(idx_bdy_beta),
                               dtype=jnp.int64)

        idx_a_1 = jnp.setdiff1d(idx_a_1, idx_a_3)
        idx_b_2 = jnp.setdiff1d(idx_b_2, idx_b_3)
        
        # extract the operators
        # in theory these operators need to be erased
        T_tau_alpha = child_alpha.T_tau
        T_tau_beta = child_beta.T_tau

        # partition the operators
        # first for alpha
        T_alpha_11 = T_tau_alpha[idx_a_1,:][:,idx_a_1]
        T_alpha_13 = T_tau_alpha[idx_a_1,:][:,idx_a_3]

        T_alpha_31 = T_tau_alpha[idx_a_3,:][:,idx_a_1]
        T_alpha_33 = T_tau_alpha[idx_a_3,:][:,idx_a_3]

        # then for beta
        T_beta_22 = T_tau_beta[idx_b_2,:][:,idx_b_2]
        T_beta_23 = T_tau_beta[idx_b_2,:][:,idx_b_3]

        T_beta_32 = T_tau_beta[idx_b_3,:][:,idx_b_2]
        T_beta_33 = T_tau_beta[idx_b_3,:][:,idx_b_3]
    
        # we extract the dimension for the block diagonal matrix
        m_alpha, n_alpha = T_alpha_11.shape
        m_beta,  n_beta  = T_beta_22.shape

        # computing the operators for the box
        self.X_tau = jnp.linalg.inv(T_alpha_33 - T_beta_33)
        self.H_tau = jnp.vstack([ T_alpha_13, T_beta_23]) @ self.X_tau

        self.S_tau = self.X_tau @ jnp.hstack([-T_alpha_31, T_beta_32])
        self.T_tau = jnp.block([[T_alpha_11, jnp.zeros((m_alpha, n_beta))],\
                                [jnp.zeros((m_beta, n_alpha)), T_beta_22]])\
                   + jnp.vstack([ T_alpha_13, T_beta_23]) @ self.S_tau

        # computing the new grid   
        # here is a bug! 
        self.x_grid = jnp.hstack( [x_grid_alpha[idx_bdy_alpha][idx_a_1],
                                   x_grid_beta[idx_bdy_beta][idx_b_2]])
        self.y_grid = jnp.hstack( [y_grid_alpha[idx_bdy_alpha][idx_a_1],
                                   y_grid_beta[idx_bdy_beta][idx_b_2]])                           
                                  
        # and the indices
        assert m_alpha + m_beta == len(self.x_grid)

        idx_bdy   = jnp.arange(0, len(self.x_grid))
        idx_bdy_1 = jnp.arange(0, m_alpha)
        idx_bdy_2 = jnp.arange(m_alpha, len(self.x_grid))

        self.idx_bdy = (idx_bdy, idx_bdy_1, idx_bdy_2)

        # we save the indices for the childrens
        self.idx_a_1 = idx_a_1
        self.idx_a_3 = idx_a_3

        self.idx_b_2 = idx_b_2
        self.idx_b_3 = idx_b_3

    def solve(self, u_bdy):
        "function to solve u_i fro u_bdy"

        assert u_bdy.shape[0] == self.S_tau.shape[1]

        # solve for the interior degrees of freedom
        u_i = self.S_tau @ u_bdy

        # we return the solution for alpha and beta

        u_alpha = jnp.zeros(((len(self.idx_a_1)+len(self.idx_a_3)),))
        u_beta  = jnp.zeros(((len(self.idx_b_2)+len(self.idx_b_3)),))

        u_alpha = u_alpha.at[self.idx_a_1].set(u_bdy[self.idx_bdy[1]])
        u_alpha = u_alpha.at[self.idx_a_3].set(u_i) 

        u_beta = u_beta.at[self.idx_b_2].set(u_bdy[self.idx_bdy[2]])
        u_beta = u_beta.at[self.idx_b_3].set(u_i) 

        return (u_alpha, u_beta)

    # def solve_root_load(self, f_bdy, h_bdy):
    #     "function to solve u_i fro u_bdy"

    #     assert f_bdy.shape[0] == self.T_tau.shape[1]

    #     # solve for the interior degrees of freedom
    #     u_bdy = self.T_tau @ f_bdy.reshape((-1,))

    #     # the call the solve routine
    #     return self.solve(u_bdy)
    
    def solve_load_up(self, h_alpha, h_beta):
        "function to solve u_i fro u_bdy"

        # spliting the load in interior and boundary
        h_alpha_1 = h_alpha[self.idx_a_1]
        h_alpha_3 = h_alpha[self.idx_a_3]
        
        h_beta_2 = h_beta[self.idx_b_2]
        h_beta_3 = h_beta[self.idx_b_3]

        delta_h_3 = (-h_alpha_3 + h_beta_3)
        # computing the interior solution
        w_int = self.X_tau @ delta_h_3

        # computing the exterior fluxes
        h_bdy = jnp.hstack([h_alpha_1, h_beta_2]) + self.H_tau @ delta_h_3

        return (w_int, h_bdy)

    def solve_load_down(self, w_int, u_bdy):
        "function to solve u_i fro u_bdy"

        # we call the regular solve
        (u_alpha, u_beta) = self.solve(u_bdy)
        
        u_alpha = u_alpha.at[self.idx_a_3].add(w_int)
        u_beta  = u_beta.at[self.idx_b_3].add(w_int)

        return (u_alpha, u_beta)


def solve_up_one_level(box_lvl_1, box_lvl_2, h_bdy_lvl_0 ):
    # function to solve the merge we assume that the first merge is in the 
    # y direction, and that the second is in the x direction

    # first level
    m_lvl_1 = len(box_lvl_1)
    n_lvl_1 = len(box_lvl_1[0])

    # merge in the x direction
    w_int_lvl_1 = jnp.zeros((m_lvl_1, n_lvl_1, box_lvl_1[0][0].X_tau.shape[0]))
    h_bdy_lvl_1 = jnp.zeros((m_lvl_1, n_lvl_1, box_lvl_1[0][0].H_tau.shape[0]))

    for i in range(m_lvl_1):
        for j in range(n_lvl_1):
            w, h = box_lvl_1[i][j].solve_load_up(h_bdy_lvl_0[i,2*j,:],
                                                 h_bdy_lvl_0[i,2*j+1,:])
            w_int_lvl_1 = w_int_lvl_1.at[i,j,:].set(w)
            h_bdy_lvl_1 = h_bdy_lvl_1.at[i,j,:].set(h)

    # merge in the y direction
    m_lvl_2 = len(box_lvl_2)
    n_lvl_2 = len(box_lvl_2[0])

    w_int_lvl_2 = jnp.zeros((m_lvl_2, n_lvl_2, box_lvl_2[0][0].X_tau.shape[0]))
    h_bdy_lvl_2 = jnp.zeros((m_lvl_2, n_lvl_2, box_lvl_2[0][0].H_tau.shape[0]))

    for i in range(m_lvl_2):
        for j in range(n_lvl_2):
            w, h = box_lvl_2[i][j].solve_load_up(h_bdy_lvl_1[2*i  ,j,:],
                                                 h_bdy_lvl_1[2*i+1,j,:])
            w_int_lvl_2 = w_int_lvl_2.at[i,j,:].set(w)
            h_bdy_lvl_2 = h_bdy_lvl_2.at[i,j,:].set(h)

    return ((w_int_lvl_1, w_int_lvl_2), 
            (h_bdy_lvl_1, h_bdy_lvl_2))


def solve_down_one_level(box_lvl_0, box_lvl_1, box_lvl_2, w_int_lvl_1, w_int_lvl_2, u_bdy_lvl_2):
    # function to solve the merge we assume that the first merge is in the 
    # y direction, and that the second is in the x direction

    m_lvl_2 = len(box_lvl_2)
    n_lvl_2 = len(box_lvl_2[0])

    m_lvl_1 = len(box_lvl_1)
    n_lvl_1 = len(box_lvl_1[0])

    m_lvl_0 = len(box_lvl_0)
    n_lvl_0 = len(box_lvl_0[0])

    u_bdy_lvl_1 = jnp.zeros((m_lvl_1, n_lvl_1, box_lvl_1[0][0].H_tau.shape[0]))

    for i in range(m_lvl_2):
        for j in range(n_lvl_2):
            u_bdy_alpha,\
            u_bdy_beta = box_lvl_2[i][j].solve_load_down(w_int_lvl_2[i,j,:],
                                                         u_bdy_lvl_2[i,j,:])
            u_bdy_lvl_1 = u_bdy_lvl_1.at[2*i  ,j,:].set(u_bdy_alpha)
            u_bdy_lvl_1 = u_bdy_lvl_1.at[2*i+1,j,:].set(u_bdy_beta)

    u_bdy_lvl_0 = jnp.zeros((m_lvl_0, n_lvl_0, box_lvl_0[0][0].H_tau.shape[0]))

    for i in range(m_lvl_1):
        for j in range(n_lvl_1):
            u_bdy_alpha,\
            u_bdy_beta = box_lvl_1[i][j].solve_load_down(w_int_lvl_1[i,j,:],
                                                         u_bdy_lvl_1[i,j,:])
            u_bdy_lvl_0 = u_bdy_lvl_0.at[i, 2*j  ,:].set(u_bdy_alpha)
            u_bdy_lvl_0 = u_bdy_lvl_0.at[i, 2*j+1,:].set(u_bdy_beta)

    return u_bdy_lvl_0

def solve_int(box_lvl_0, w_int_lvl_0, u_bdy_lvl_0):

    # extracting the dimensions of the array
    m_lvl_0 = len(box_lvl_0)
    n_lvl_0 = len(box_lvl_0[0])
        
    # defining the interior degrees of freedom
    u_int = jnp.zeros((m_lvl_0, n_lvl_0, box_lvl_0[0][0].X_tau.shape[0]))


    for i in range(m_lvl_0):
        for j in range(n_lvl_0):
            u_int_loc = box_lvl_0[i][j].solve_load_down(w_int_lvl_0[i,j,:],
                                                   u_bdy_lvl_0[i,j,:])
            u_int = u_int.at[i,j,:].set(u_int_loc)
    
    return u_int


def solve_leaves(box_lvl_0, rhs_grid):

    # extractint the size of the size
    nx = box_lvl_0[0][0].x_grid.shape[0]-1 

    m_lvl_0 = len(box_lvl_0)
    n_lvl_0 = len(box_lvl_0[0])

    w_int_lvl_0 = jnp.zeros((m_lvl_0, n_lvl_0, (nx-1)**2))
    h_bdy_lvl_0 = jnp.zeros((m_lvl_0, n_lvl_0, 4*(nx-1)))

    for i in range(m_lvl_0):
        for j in range(n_lvl_0):
            w, h = box_lvl_0[i][j].solve_load_up(rhs_grid[i][j])
            w_int_lvl_0 = w_int_lvl_0.at[i,j,:].set(w)
            h_bdy_lvl_0 = h_bdy_lvl_0.at[i,j,:].set(h)

    return (w_int_lvl_0, h_bdy_lvl_0)

###############################################################################
#                           Jax Classes                                       #
###############################################################################

@dataclasses.dataclass
class Leaf:

    T_tau: Array
    H_tau: Array
    X_tau: Array
    S_tau: Array

    x_grid: Array
    y_grid: Array

    idx_int: Array
    idx_bdy: List[Array]

    # todo: add a update function

@dataclasses.dataclass
class Box:

    T_tau: Array
    H_tau: Array
    X_tau: Array
    S_tau: Array

    x_grid: Array
    y_grid: Array

    idx_bdy: List[Array] # perhaps a tuple

    idx_a_1: Array
    idx_a_3: Array

    idx_b_2: Array
    idx_b_3: Array



def init_leaf_jax(nx, ny, center):
    # function to initialize a leaf

    (operators, Diff,
    grid,
    idx_int, idx_bdy) = cheb_2d_poincare_steklov(nx, ny)


    return Leaf(operators["T_tau"], 
                operators["H_tau"],
                operators["X_tau"],
                operators["S_tau"],
                grid[0] + center[0],
                grid[1] + center[1],
                idx_int,
                idx_bdy)

# TODO: we need to create a simple scaffolind for the 
# boxes... basically we don't need to define any of the
# operators, only the indices... which will be done in python
# then the update of the operators will be done in JAX


def init_box_jax(child_alpha, child_beta):
    # this function only initiliates the box
    # it is used to start the indexation 
    # this is not is not jittable (at least not easily)

    # compute the indices in common
    idx_bdy_alpha = child_alpha.idx_bdy[0]
    idx_bdy_beta  = child_beta.idx_bdy[0]

    # extract the grids, we reshape it so they are in the 
    # correct format, this is for the leafs for which the 
    # grids fields contain the dimensions of the rectangle
    x_grid_alpha = child_alpha.x_grid.reshape((-1,))
    y_grid_alpha = child_alpha.y_grid.reshape((-1,))
    
    x_grid_beta = child_beta.x_grid.reshape((-1,))
    y_grid_beta = child_beta.y_grid.reshape((-1,))

    diff_x = x_grid_alpha[idx_bdy_alpha].reshape((-1,1))\
            - x_grid_beta[idx_bdy_beta].reshape((1,-1)) 
    
    diff_y = y_grid_alpha[idx_bdy_alpha].reshape((-1,1))\
            - y_grid_beta[idx_bdy_beta].reshape((1,-1)) 

    diff_norm = jnp.sqrt(jnp.square(diff_x) + jnp.square(diff_y))

    # we want the indices in which the two values are the same
    idx_a_3, idx_b_3 = jnp.where(jnp.isclose(diff_norm, 0.))

    # we make sure that the indices are correct, i.e. that they correspond
    # to the same grid points
    assert jnp.isclose(jnp.linalg.norm(x_grid_alpha[idx_bdy_alpha][idx_a_3]\
                                        - x_grid_beta[idx_bdy_beta][idx_b_3]),
                        0.0)
    assert jnp.isclose(jnp.linalg.norm(y_grid_alpha[idx_bdy_alpha][idx_a_3]\
                                        - y_grid_beta[idx_bdy_beta][idx_b_3]),
                        0.0)

    # split the different indices
    idx_a_1 = jnp.linspace(0, len(idx_bdy_alpha)-1, len(idx_bdy_alpha),
                            dtype=jnp.int64)
    idx_b_2 = jnp.linspace(0, len(idx_bdy_beta)-1, len(idx_bdy_beta),
                            dtype=jnp.int64)

    idx_a_1 = jnp.setdiff1d(idx_a_1, idx_a_3)
    idx_b_2 = jnp.setdiff1d(idx_b_2, idx_b_3)
    
    # we extract the dimension for the block diagonal matrix
    m_alpha = len(idx_a_1)
    m_beta = len(idx_b_2)

    # computing the new grid   
    # here is a bug! 
    x_grid = jnp.hstack( [x_grid_alpha[idx_bdy_alpha][idx_a_1],
                                x_grid_beta[idx_bdy_beta][idx_b_2]])
    y_grid = jnp.hstack( [y_grid_alpha[idx_bdy_alpha][idx_a_1],
                                y_grid_beta[idx_bdy_beta][idx_b_2]])                           
                                
    # and the indices
    assert m_alpha + m_beta == len(x_grid)

    idx_bdy   = jnp.arange(0, len(x_grid))
    idx_bdy_1 = jnp.arange(0, m_alpha)
    idx_bdy_2 = jnp.arange(m_alpha, len(x_grid))

    idx_bdy = (idx_bdy, idx_bdy_1, idx_bdy_2)

    # we save the indices for the childrens
    idx_a_1 = idx_a_1
    idx_a_3 = idx_a_3

    idx_b_2 = idx_b_2
    idx_b_3 = idx_b_3

    return Box(jnp.zeros((1,1)), jnp.zeros((1,1)),
               jnp.zeros((1,1)), jnp.zeros((1,1)),
               x_grid, y_grid, 
               idx_bdy,
               idx_a_1, idx_a_3, idx_b_2, idx_b_3)


def update_box_jax(box:Box, child_alpha, child_beta):
    # merge the two operators of the boxes, they can be either leaves
    # or other boxed

    # extract the operators
    # in theory these operators need to be erased
    T_tau_alpha = child_alpha.T_tau
    T_tau_beta  = child_beta.T_tau

    # partition the operators
    # first for alpha
    T_alpha_11 = T_tau_alpha[box.idx_a_1,:][:,box.idx_a_1]
    T_alpha_13 = T_tau_alpha[box.idx_a_1,:][:,box.idx_a_3]

    T_alpha_31 = T_tau_alpha[box.idx_a_3,:][:,box.idx_a_1]
    T_alpha_33 = T_tau_alpha[box.idx_a_3,:][:,box.idx_a_3]

    # then for beta
    T_beta_22 = T_tau_beta[box.idx_b_2,:][:,box.idx_b_2]
    T_beta_23 = T_tau_beta[box.idx_b_2,:][:,box.idx_b_3]

    T_beta_32 = T_tau_beta[box.idx_b_3,:][:,box.idx_b_2]
    T_beta_33 = T_tau_beta[box.idx_b_3,:][:,box.idx_b_3]

    # we extract the dimension for the block diagonal matrix
    m_alpha, n_alpha = T_alpha_11.shape
    m_beta,  n_beta  = T_beta_22.shape

    # computing the operators for the box
    X_tau = jnp.linalg.inv(T_alpha_33 - T_beta_33)
    H_tau = jnp.vstack([ T_alpha_13, T_beta_23]) @ X_tau

    S_tau = X_tau @ jnp.hstack([-T_alpha_31, T_beta_32])
    T_tau = jnp.block([[T_alpha_11, jnp.zeros((m_alpha, n_beta))],\
                            [jnp.zeros((m_beta, n_alpha)), T_beta_22]])\
                + jnp.vstack([ T_alpha_13, T_beta_23]) @ S_tau


    return box.replace(T_tau=T_tau, H_tau=H_tau, 
                       X_tau=X_tau, S_tau=S_tau)


def solve_leaf(leaf: Leaf, u_bdy: Array):
    return leaf.S_tau @ u_bdy

def solve_box(box: Box, u_bdy: Array):
    "function to solve u_i fro u_bdy"

    assert u_bdy.shape[0] == box.S_tau.shape[1]

    # solve for the interior degrees of freedom
    u_i = box.S_tau @ u_bdy

    # we return the solution for alpha and beta

    u_alpha = jnp.zeros(((len(box.idx_a_1)+len(box.idx_a_3)),))
    u_beta  = jnp.zeros(((len(box.idx_b_2)+len(box.idx_b_3)),))

    u_alpha = u_alpha.at[box.idx_a_1].set(u_bdy[box.idx_bdy[1]])
    u_alpha = u_alpha.at[box.idx_a_3].set(u_i) 

    u_beta = u_beta.at[box.idx_b_2].set(u_bdy[box.idx_bdy[2]])
    u_beta = u_beta.at[box.idx_b_3].set(u_i) 

    return (u_alpha, u_beta)

def solve_leaf_load_up(leaf: Leaf, g: Array):
    # computing for the load
    g_int = g[leaf.idx_int]
    # compute derivatives of the local solution
    h_bdy = leaf.H_tau @ g_int
    # compute the interior of the load problem
    w_int = leaf.X_tau @ g_int

    return (w_int, h_bdy)

def solve_leaf_load_down(leaf: Leaf, w_int: Array, u_bdy: Array):
    return solve_leaf(leaf, u_bdy) + w_int

def solve_box_load_up(box: Box, h_alpha: Array, h_beta: Array):
    "function to solve u_i fro u_bdy"

    # spliting the load in interior and boundary
    h_alpha_1 = h_alpha[box.idx_a_1]
    h_alpha_3 = h_alpha[box.idx_a_3]
    
    h_beta_2 = h_beta[box.idx_b_2]
    h_beta_3 = h_beta[box.idx_b_3]

    delta_h_3 = (-h_alpha_3 + h_beta_3)
    # computing the interior solution
    w_int = box.X_tau @ delta_h_3

    # computing the exterior fluxes
    h_bdy = jnp.hstack([h_alpha_1, h_beta_2]) + box.H_tau @ delta_h_3

    return (w_int, h_bdy)

def solve_box_load_down(box: Box, w_int: Array, u_bdy: Array):
    "function to solve u_i fro u_bdy"

    # we call the regular solve
    (u_alpha, u_beta) = solve_box(box, u_bdy)
    
    u_alpha = u_alpha.at[box.idx_a_3].add(w_int)
    u_beta  = u_beta.at[box.idx_b_3].add(w_int)

    return (u_alpha, u_beta)

###############################################################################
#             functions working on the arrays of boxes and leaves             #
###############################################################################

def solve_leaves_jax(box_lvl_0, rhs_grid):
    # solving the leaves
    # we may need to jit and vectorize this

    # extractint the size of the size
    nx = box_lvl_0[0][0].x_grid.shape[0]-1 

    m_lvl_0 = len(box_lvl_0)
    n_lvl_0 = len(box_lvl_0[0])

    w_int_lvl_0 = jnp.zeros((m_lvl_0, n_lvl_0, (nx-1)**2))
    h_bdy_lvl_0 = jnp.zeros((m_lvl_0, n_lvl_0, 4*(nx-1)))

    for i in range(m_lvl_0):
        for j in range(n_lvl_0):
            w, h = solve_leaf_load_up(box_lvl_0[i][j],rhs_grid[i][j])
            w_int_lvl_0 = w_int_lvl_0.at[i,j,:].set(w)
            h_bdy_lvl_0 = h_bdy_lvl_0.at[i,j,:].set(h)

    return (w_int_lvl_0, h_bdy_lvl_0)


def update_box_one_level(box_lvl_0, box_lvl_1, box_lvl_2):
    # function to update the boxes at a certain level

    for i in range(len(box_lvl_1)):
        for j in range(len(box_lvl_1[0])):
            box_lvl_1[i][j] = update_box_jax(box_lvl_1[i][j], 
                                             box_lvl_0[i][2*j],
                                             box_lvl_0[i][2*j+1])

    for i in range(len(box_lvl_2)):
        for j in range(len(box_lvl_2[0])):
            box_lvl_2[i][j] = update_box_jax(box_lvl_2[i][j], 
                                             box_lvl_1[2*i][j],
                                             box_lvl_1[2*i+1][j])

    return (box_lvl_1, box_lvl_2)


def solve_up_one_level_jax(box_lvl_1, box_lvl_2, h_bdy_lvl_0 ):
    # function to solve the merge we assume that the first merge is in the 
    # y direction, and that the second is in the x direction

    # first level
    m_lvl_1 = len(box_lvl_1)
    n_lvl_1 = len(box_lvl_1[0])

    # merge in the x direction
    w_int_lvl_1 = jnp.zeros((m_lvl_1, n_lvl_1, box_lvl_1[0][0].X_tau.shape[0]))
    h_bdy_lvl_1 = jnp.zeros((m_lvl_1, n_lvl_1, box_lvl_1[0][0].H_tau.shape[0]))

    for i in range(m_lvl_1):
        for j in range(n_lvl_1):
            w, h = solve_box_load_up(box_lvl_1[i][j], 
                                     h_bdy_lvl_0[i,2*j,:],
                                     h_bdy_lvl_0[i,2*j+1,:])
            w_int_lvl_1 = w_int_lvl_1.at[i,j,:].set(w)
            h_bdy_lvl_1 = h_bdy_lvl_1.at[i,j,:].set(h)

    # merge in the y direction
    m_lvl_2 = len(box_lvl_2)
    n_lvl_2 = len(box_lvl_2[0])

    w_int_lvl_2 = jnp.zeros((m_lvl_2, n_lvl_2, box_lvl_2[0][0].X_tau.shape[0]))
    h_bdy_lvl_2 = jnp.zeros((m_lvl_2, n_lvl_2, box_lvl_2[0][0].H_tau.shape[0]))

    for i in range(m_lvl_2):
        for j in range(n_lvl_2):
            w, h = solve_box_load_up(box_lvl_2[i][j],
                                     h_bdy_lvl_1[2*i  ,j,:],
                                     h_bdy_lvl_1[2*i+1,j,:])
            w_int_lvl_2 = w_int_lvl_2.at[i,j,:].set(w)
            h_bdy_lvl_2 = h_bdy_lvl_2.at[i,j,:].set(h)

    return ((w_int_lvl_1, w_int_lvl_2), 
            (h_bdy_lvl_1, h_bdy_lvl_2))


def solve_down_one_level_jax(box_lvl_0, box_lvl_1, box_lvl_2, 
                             w_int_lvl_1, w_int_lvl_2, u_bdy_lvl_2):
    # function to solve the merge we assume that the first merge is in the 
    # y direction, and that the second is in the x direction

    m_lvl_2 = len(box_lvl_2)
    n_lvl_2 = len(box_lvl_2[0])

    m_lvl_1 = len(box_lvl_1)
    n_lvl_1 = len(box_lvl_1[0])

    m_lvl_0 = len(box_lvl_0)
    n_lvl_0 = len(box_lvl_0[0])

    u_bdy_lvl_1 = jnp.zeros((m_lvl_1, n_lvl_1, box_lvl_1[0][0].H_tau.shape[0]))

    for i in range(m_lvl_2):
        for j in range(n_lvl_2):
            u_bdy_alpha,\
            u_bdy_beta = solve_box_load_down(box_lvl_2[i][j],
                                             w_int_lvl_2[i,j,:],
                                             u_bdy_lvl_2[i,j,:])
            u_bdy_lvl_1 = u_bdy_lvl_1.at[2*i  ,j,:].set(u_bdy_alpha)
            u_bdy_lvl_1 = u_bdy_lvl_1.at[2*i+1,j,:].set(u_bdy_beta)

    u_bdy_lvl_0 = jnp.zeros((m_lvl_0, n_lvl_0, box_lvl_0[0][0].H_tau.shape[0]))

    for i in range(m_lvl_1):
        for j in range(n_lvl_1):
            u_bdy_alpha,\
            u_bdy_beta = solve_box_load_down(box_lvl_1[i][j],
                                             w_int_lvl_1[i,j,:],
                                             u_bdy_lvl_1[i,j,:])
            u_bdy_lvl_0 = u_bdy_lvl_0.at[i, 2*j  ,:].set(u_bdy_alpha)
            u_bdy_lvl_0 = u_bdy_lvl_0.at[i, 2*j+1,:].set(u_bdy_beta)

    return u_bdy_lvl_0


def solve_int_jax(box_lvl_0, w_int_lvl_0, u_bdy_lvl_0):

    # extracting the dimensions of the array
    m_lvl_0 = len(box_lvl_0)
    n_lvl_0 = len(box_lvl_0[0])
        
    # defining the interior degrees of freedom
    u_int = jnp.zeros((m_lvl_0, n_lvl_0, box_lvl_0[0][0].X_tau.shape[0]))

    # looping over the leaves
    for i in range(m_lvl_0):
        for j in range(n_lvl_0):
            u_int_loc = solve_leaf_load_down(box_lvl_0[i][j], 
                                             w_int_lvl_0[i,j,:],
                                             u_bdy_lvl_0[i,j,:])
            u_int = u_int.at[i,j,:].set(u_int_loc)
    
    return u_int


#####
# functions to build the discretization 
######

def create_discretization(domain, levels):
    min_x, max_x = domain[0]
    min_y, max_y = domain[1]

    num_domains = 2**levels


