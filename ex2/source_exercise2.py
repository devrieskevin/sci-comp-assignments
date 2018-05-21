"""
Author:         Kevin de Vries
Student number: 10579869

This file contains the source code for the implementations of
the methods used for the exercises in exercise set 2.

Builds on functions implemented in the source file used
for exercise set 1.
"""

import numpy as np
import tqdm
from scipy.misc import imread,imresize

class object_grid(object):
    """
    An object grid class which can be used to maintain a grid of objects,
    which can be used to dynamically add objects to the same grid.
    """

    def __init__(self,nx,ny,dx=None,dy=None):
        self.grid = np.zeros((nx,ny))
        self.dx = dx
        self.dy = dy

    def set_grid_spacing(self,dx,dy):
        "Set the grid spacing"

        self.dx = dx
        self.dy = dy

    def load_image(self,filename):
        "Load and resize an image into the grid"

        imarray = imresize(imread(filename,flatten=True),self.grid.shape)
        imbools = (1 - imarray / 255) > 0
        self.grid[imbools] = 1

    def load_rectangle(self,imin,imax,jmin,jmax):
        "Load a rectangular object into the grid using given indices"

        self.grid[imin:imax+1,jmin:jmax+1] = 1

    def load_rectangle_real(self,xmin,xmax,ymin,ymax):
        "Load a rectangular objects into the grid using real valued coordinates"

        imin = int(xmin / self.dx)
        imax = int(xmax / self.dx)
        jmin = int(ymin / self.dy)
        jmax = int(ymax / self.dy)
        self.load_object(imin,imax,jmin,jmax)

    def set_bool_array(self,array):
        self.grid[array] = 1

    def get_growth_candidates(self):
        """
        Returns a list with growth candidates
        """
        nx,ny = self.grid.shape
        res = np.empty((nx,ny),dtype=bool)

        bools = self.get_boolean_grid()
        res[1:nx-1,1:ny-1] = bools[0:nx-2,1:ny-1] | bools[2:nx,1:ny-1] | \
                             bools[1:nx-1,0:ny-2] | bools[1:nx-1,2:ny]
        
        res[0,1:ny-1] = bools[1,1:ny-1] | bools[nx-2,1:ny-1] | \
                        bools[0,0:ny-2] | bools[0,2:ny]

        res[nx-1,1:ny-1] = False
        res[:,0] = res[:,ny-1] = False

        return res & ~bools

    def get_boolean_grid(self):
        "Returns a boolean array with object positions"

        return self.grid != 0

def solve_laplace_SOR(N,eps,init_c,bound_params,misc_params,w=1,bool_grid=None,vectorized=True):
    """
    Solves the time independent diffusion equation using Successive Over Relaxation iteration
    with periodic boundary conditions in x and dirichlet boundary conditions in y.
    The relaxation factor w is set to 1 by default, which yields a Gauss-Seidel iteration.
    """

    # boundary condition functions
    bound_y0, bound_yN = bound_params
    # diffusion equation parameters
    xmin,xmax = misc_params

    # step size in x and y
    dx = (xmax-xmin) / N

    c = np.empty((2,N+1,N+1))

    # initialize the values of c
    c[1,0:N+1,1:N] = init_c(dx*np.arange(0,N+1),dx*np.arange(1,N))

    # insert the boundary conditions
    c[1,:,0] = bound_y0(dx*np.arange(0,N))
    c[1,:,N] = bound_yN(dx*np.arange(0,N))

    deltas = []

    if bool_grid is not None:
        c[1][bool_grid] = 0
        c[1,N,1:N:2] = c[1,0,1:N:2]

    while True:
        # set old values
        c[0,:,:] = c[1,:,:]

        if vectorized:
            # vectorized implementation with red-black ordering

            # update red points
            c[1,2:N:2,1:N:2] = (1 - w) * c[1,2:N:2,1:N:2] + 0.25 * w * \
                               (c[1,1:N-1:2,1:N:2] + c[1,3:N+1:2,1:N:2] + c[1,2:N:2,0:N-1:2] + c[1,2:N:2,2:N+1:2])

            c[1,1:N:2,2:N:2] = (1 - w) * c[1,1:N:2,2:N:2] + 0.25 * w * \
                               (c[1,0:N-1:2,2:N:2] + c[1,2:N+1:2,2:N:2] + c[1,1:N:2,1:N-1:2] + c[1,1:N:2,3:N+1:2])

            # update red periodic boundaries
            c[1,N,1:N:2] = c[1,0,1:N:2] = (1 - w) * c[1,0,1:N:2] + 0.25 * w * \
                                          (c[1,N-1,1:N:2] + c[1,1,1:N:2] + c[1,0,0:N-1:2] + c[1,0,2:N+1:2])

            if bool_grid is not None:
                c[1][bool_grid] = 0
                c[1,N,1:N:2] = c[1,0,1:N:2]

            #update black points
            c[1,2:N:2,2:N:2] = (1 - w) * c[1,2:N:2,2:N:2] + 0.25 * w * \
                               (c[1,1:N-1:2,2:N:2] + c[1,3:N+1:2,2:N:2] + c[1,2:N:2,1:N-1:2] + c[1,2:N:2,3:N+1:2])
            
            c[1,1:N:2,1:N:2] = (1 - w) * c[1,1:N:2,1:N:2] + 0.25 * w * \
                               (c[1,0:N-1:2,1:N:2] + c[1,2:N+1:2,1:N:2] + c[1,1:N:2,0:N-1:2] + c[1,1:N:2,2:N+1:2])

            # update black periodic boundaries
            c[1,N,2:N:2] = c[1,0,2:N:2] = (1 - w) * c[1,0,2:N:2] + 0.25 * w * \
                                          (c[1,N-1,2:N:2] + c[1,1,2:N:2] + c[1,0,1:N-1:2] + c[1,0,3:N+1:2])

            if bool_grid is not None:
                c[1][bool_grid] = 0
                c[1,N,1:N:2] = c[1,0,1:N:2]

        else:
            # update with old values
            c[1,0:N,1:N] = (1 - w) * c[0,0:N,1:N] + 0.25 * w * (c[0,0:N,2:N+1] + c[0,1:N+1,1:N])
            for j in range(1,N):
                # update with each last future row
                c[1,0:N,j] += 0.25 * w * c[1,0:N,j-1]

                # update with each left future value
                c[1,0,j] += 0.25 * w * c[0,N-1,j]
                for i in range(1,N):
                    c[1,i,j] += 0.25 * w * c[1,i-1,j]

            # update periodic boundaries in x
            c[1,N,1:N] = c[1,0,1:N]

            if bool_grid is not None:
                c[1][bool_grid] = 0
                c[1,N,1:N:2] = c[1,0,1:N:2]

        delta = np.amax(np.abs(c[1,:,:]-c[0,:,:]))
        deltas.append(delta)
        if delta < eps:
            return c[1,:,:],deltas

def DLA_pde(n_iter,N,eps,init_c,bound_params,misc_params,init_grid,w=1,eta=1):
    """
    Performs a growth simulation using the Diffusion Limited Aggregation
    growth model. The implementation is done using a numeric PDE solver.
    """

    grid = init_grid
    init_func = init_c

    with tqdm.tqdm(total=n_iter) as pbar:
        for i in range(n_iter):
            bool_grid = grid.get_boolean_grid()
            c, deltas = solve_laplace_SOR(N,eps,init_func,bound_params,misc_params,w,bool_grid)

            # calculate probabilities
            candidates = grid.get_growth_candidates()
            c_candidates = c[candidates]
            c_candidates[c_candidates < 0] = 0
            probs = c_candidates**eta / np.sum(c_candidates**eta)

            # Determine growth sites
            rand_nums = np.random.uniform(0,1,probs.size)
            candidates[candidates] = rand_nums < probs

            # Load set growth into the grid
            grid.set_bool_array(candidates)

            init_func = lambda x,y: c[:,1:N]

            pbar.update()

    return c,grid

def DLA_MC(c_size,N,init_grid,ps=1):
    """
    Performs a growth simulation using the Diffusion Limited Aggregation
    growth model. The implementation is done using Monte Carlo simulation.
    """

    grid = init_grid
    neumann = np.array([[-1,0],[1,0],[0,-1],[0,1]])
    
    with tqdm.tqdm(total=c_size) as pbar:
        for i in range(c_size):
            # walker coordinates
            coords = np.array([np.random.randint(0,N),N])
        
            # Get the boolean object and growth candidate arrays
            bool_grid = grid.get_boolean_grid()
            growth_sites = grid.get_growth_candidates()
        
            stuck = False
            while not stuck:
                rand_num = np.random.randint(4)

                # Randomly choose new coordinate
                new_coords = coords + neumann[rand_num]
            
                # Reset coordinates if out of bounds
                if new_coords[1] < 0 or new_coords[1] > N:
                    coords = np.array([np.random.randint(0,N),N])
                    continue
            
                # Enforce periodic boundaries
                new_coords[0] = new_coords[0] % N
            
                x,y = new_coords
            
                # Try again if overlapping with object
                if bool_grid[x,y]:
                    continue
            
                coords = new_coords
            
                # Determine if stuck when on candidate site
                if growth_sites[x,y]:
                    rand_num = np.random.uniform()
                    if rand_num < ps:
                        grid.load_rectangle(x,x,y,y)
                        stuck = True
    
            pbar.update()
    
    return grid

def gray_scott_periodic(step_params, init_params, misc_params):
    """
    Numerically solves the reaction-diffusion equations of the Gray-Scott model
    using periodic boundary conditions in both the x and y directions.
    """

    # step parameters
    N,M,dx,dt = step_params
    # initial condition functions
    init_u, init_v = init_params
    # wave equation parameters
    f,k,Du,Dv = misc_params

    u = np.empty((M+1,N+1,N+1))
    v = np.empty((M+1,N+1,N+1))
    
    # initialize the values of u and v
    u[0,0:N,0:N] = init_u(dx*np.arange(0,N),dx*np.arange(0,N))
    v[0,0:N,0:N] = init_v(dx*np.arange(0,N),dx*np.arange(0,N))

    # Enforce periodic boundary conditions on initial conditions
    u[0,N,:] = u[0,0,:]
    u[0,:,N] = u[0,:,0]
    v[0,N,:] = v[0,0,:]
    v[0,:,N] = v[0,:,0]

    # Prefactor coefficients
    cu = dt * Du / dx**2
    cv = dt * Dv / dx**2
    
    for m in range(1,M+1):
        uv2 = u[m-1,0:N,0:N] * v[m-1,0:N,0:N]**2
        
        # update normal u points
        u[m,1:N,1:N] = (1 - 4*cu) * u[m-1,1:N,1:N]
        u[m,1:N,1:N] += cu * (u[m-1,2:N+1,1:N] + u[m-1,0:N-1,1:N] + u[m-1,1:N,2:N+1] + u[m-1,1:N,0:N-1])

        # update periodic boundaries of u in x
        u[m,0,1:N] = (1 - 4*cu) * u[m-1,0,1:N]
        u[m,0,1:N] += cu * (u[m-1,1,1:N] + u[m-1,N-1,1:N] + u[m-1,0,2:N+1] + u[m-1,0,0:N-1])

        # update periodic boundaries of u in y
        u[m,1:N,0] = (1 - 4*cu) * u[m-1,1:N,0]
        u[m,1:N,0] += cu * (u[m-1,2:N+1,0] + u[m-1,0:N-1,0] + u[m-1,1:N,1] + u[m-1,1:N,N-1])
        
        # update index (0,0)
        u[m,0,0] = (1 - 4*cu) * u[m-1,0,0]
        u[m,0,0] += cu * (u[m-1,1,0] + u[m-1,N-1,0] + u[m-1,0,1] + u[m-1,0,N-1])

        # Reaction terms
        u[m,0:N,0:N] += dt * (f * (1 - u[m-1,0:N,0:N]) - uv2)
        
        # Synchronize periodic points
        u[m,N,:] = u[m,0,:]
        u[m,:,N] = u[m,:,0]
        
        # update normal v points
        v[m,1:N,1:N] = (1 - 4*cv) * v[m-1,1:N,1:N]
        v[m,1:N,1:N] += cv * (v[m-1,2:N+1,1:N] + v[m-1,0:N-1,1:N] + v[m-1,1:N,2:N+1] + v[m-1,1:N,0:N-1])

        # update periodic boundaries of v in x
        v[m,0,1:N] = (1 - 4*cv) * v[m-1,0,1:N]
        v[m,0,1:N] += cv * (v[m-1,1,1:N] + v[m-1,N-1,1:N] + v[m-1,0,2:N+1] + v[m-1,0,0:N-1])

        # update periodic boundaries of v in y
        v[m,1:N,0] = (1 - 4*cv) * v[m-1,1:N,0]
        v[m,1:N,0] += cv * (v[m-1,2:N+1,0] + v[m-1,0:N-1,0] + v[m-1,1:N,1] + v[m-1,1:N,N-1])
        
        # update index (0,0)
        v[m,0,0] = (1 - 4*cv) * v[m-1,0,0]
        v[m,0,0] += cv * (v[m-1,1,0] + v[m-1,N-1,0] + v[m-1,0,1] + v[m-1,0,N-1])

        # Reaction terms
        v[m,0:N,0:N] += dt * (uv2 - (f + k) * v[m-1,0:N,0:N])
        
        # Synchronize periodic points
        v[m,N,:] = v[m,0,:]
        v[m,:,N] = v[m,:,0]
    
    return u,v

def gray_scott_neumann(step_params, init_params, bound_params, misc_params):
    """
    Numerically solves the reaction-diffusion equations of the Gray-Scott model
    using neumann boundary conditions in both the x and y directions.

    For the sake of simplicity we take the boundary conditions to be constant
    over the whole boundary specified for both reactants
    """

    # step parameters
    N,M,dx,dt = step_params
    # initial condition functions
    init_u, init_v = init_params
    # boundary condition parameters
    bound_u, bound_v = bound_params
    # wave equation parameters
    f,k,Du,Dv = misc_params

    u = np.empty((M+1,N+1,N+1))
    v = np.empty((M+1,N+1,N+1))
    
    # initialize the values of u and v
    u[0,0:N+1,0:N+1] = init_u(dx*np.arange(0,N+1),dx*np.arange(0,N+1))
    v[0,0:N+1,0:N+1] = init_v(dx*np.arange(0,N+1),dx*np.arange(0,N+1))

    # Prefactor coefficients
    cu = dt * Du / dx**2
    cv = dt * Dv / dx**2
    
    for m in range(1,M+1):
        uv2 = u[m-1,:,:] * v[m-1,:,:]**2
        
        # update normal u points
        u[m,1:N,1:N] = (1 - 4*cu) * u[m-1,1:N,1:N]
        u[m,1:N,1:N] += cu * (u[m-1,2:N+1,1:N] + u[m-1,0:N-1,1:N] + u[m-1,1:N,2:N+1] + u[m-1,1:N,0:N-1])

        # update boundary u points
        u[m,0,:] = (1 - 4*cu) * u[m-1,0,:]
        u[m,N,:] = (1 - 4*cu) * u[m-1,N,:]
        u[m,:,0] = (1 - 4*cu) * u[m-1,:,0]
        u[m,:,N] = (1 - 4*cu) * u[m-1,:,N]

        u[m,0,1:N] += cu * (2*u[m-1,1,1:N] + u[m-1,0,2:N+1] + u[m-1,0,0:N-1] - 2*dx*bound_u)
        u[m,N,1:N] += cu * (2*u[m-1,N-1,1:N] + u[m-1,N,2:N+1] + u[m-1,N,0:N-1] + 2*dx*bound_u)
        u[m,1:N,0] += cu * (u[m-1,2:N+1,0] + u[m-1,0:N-1,0] + 2*u[m-1,1:N,1] - 2*dx*bound_u)
        u[m,1:N,N] += cu * (u[m-1,2:N+1,N] + u[m-1,0:N-1,N] + 2*u[m-1,1:N,N-1] + 2*dx*bound_u)

        # update corners
        u[m,0,0] += cu * (2*u[m-1,1,0] + 2*u[m-1,0,1] - 4*dx*bound_u)
        u[m,N,0] += cu * (2*u[m-1,N-1,0] + 2*u[m-1,N,1])
        u[m,0,N] += cu * (2*u[m-1,1,N] + 2*u[m-1,0,N-1])
        u[m,N,N] += cu * (2*u[m-1,N-1,N] + 2*u[m-1,N,N-1] + 4*dx*bound_u)

        # Reaction terms
        u[m,:,:] += dt * (f * (1 - u[m-1,:,:]) - uv2)
        
        # update normal v points
        v[m,1:N,1:N] = (1 - 4*cv) * v[m-1,1:N,1:N]
        v[m,1:N,1:N] += cv * (v[m-1,2:N+1,1:N] + v[m-1,0:N-1,1:N] + v[m-1,1:N,2:N+1] + v[m-1,1:N,0:N-1])

        # update boundary v points
        v[m,0,:] = (1 - 4*cv) * v[m-1,0,:]
        v[m,N,:] = (1 - 4*cv) * v[m-1,N,:]
        v[m,:,0] = (1 - 4*cv) * v[m-1,:,0]
        v[m,:,N] = (1 - 4*cv) * v[m-1,:,N]

        v[m,0,1:N] += cv * (2*v[m-1,1,1:N] + v[m-1,0,2:N+1] + v[m-1,0,0:N-1] - 2*dx*bound_v)
        v[m,N,1:N] += cv * (2*v[m-1,N-1,1:N] + v[m-1,N,2:N+1] + v[m-1,N,0:N-1] + 2*dx*bound_v)
        v[m,1:N,0] += cv * (v[m-1,2:N+1,0] + v[m-1,0:N-1,0] + 2*v[m-1,1:N,1] - 2*dx*bound_v)
        v[m,1:N,N] += cv * (v[m-1,2:N+1,N] + v[m-1,0:N-1,N] + 2*v[m-1,1:N,N-1] + 2*dx*bound_v)

        # update corners
        v[m,0,0] += cv * (2*v[m-1,1,0] + 2*v[m-1,0,1] - 4*dx*bound_v)
        v[m,N,0] += cv * (2*v[m-1,N-1,0] + 2*v[m-1,N,1])
        v[m,0,N] += cv * (2*v[m-1,1,N] + 2*v[m-1,0,N-1])
        v[m,N,N] += cv * (2*v[m-1,N-1,N] + 2*v[m-1,N,N-1] + 4*dx*bound_v)

        # Reaction terms
        v[m,:,:] += dt * (uv2 - (f + k) * v[m-1,:,:])
            
    return u,v
