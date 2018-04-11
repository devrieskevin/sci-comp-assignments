"""
Author:         Kevin de Vries
Student number: 10579869

This file contains the source code for the implementations of
the methods used for the exercises in exercise set 1.
"""

import numpy as np
from scipy.special import erfc
from scipy.misc import imread,imresize

class object_grid(object):
    def __init__(self,nx,ny,dx=None,dy=None):
        self.grid = np.zeros((nx,ny))
        self.dx = dx
        self.dy = dy

    def set_grid_spacing(self,dx,dy):
        self.dx = dx
        self.dy = dy

    def load_image(self,filename):
        imarray = imresize(imread(filename,flatten=True),self.grid.shape)
        imbools = (1 - imarray / 255) > 0
        self.grid[imbools] = 1

    def load_rectangle(self,imin,imax,jmin,jmax):
        self.grid[imin:imax+1,jmin:jmax+1] = 1

    def load_rectangle_real(self,xmin,xmax,ymin,ymax):
        imin = int(xmin / self.dx)
        imax = int(xmax / self.dx)
        jmin = int(ymin / self.dy)
        jmax = int(ymax / self.dy)
        self.load_object(imin,imax,jmin,jmax)

    def get_boolean_grid(self):
        return self.grid != 0

def explicit_tridiagonal(coefs,vec,bound_coefs=None,bound_shift=None):
    """
    Applies the tridiagonal matrix for the 
    Finite Difference method to a given vector

    coefs: tuple-like
        contains the matrix elements of the symmetric tridiagonal matrix
    vec: numpy array
        vector to which the matrix is applied
    bound_coefs: tuple-like
        contains the matrix elements at the boundaries
    bound_shift: tuple-like
        contains the dirichlet shifts after the matrix is applied
    """

    minus,null,plus = coefs

    res = np.zeros(vec.size)

    # Calculate first and last elements
    if bound_coefs:
        u1,u2,d1,d2 = bound_coefs
        res[0] = u1*vec[0] + u2*vec[1]
        res[-1] = d1*vec[-2] + d2*vec[-1]
    else:
        res[0] = null*vec[0] + plus*vec[1]
        res[-1] = minus*vec[-2] + null*vec[-1]

    if bound_shift:
        upper,lower = bound_shift
        res[0] += upper
        res[-1] += lower


    # Calculate the rest of the vector
    res[1:-1] = minus*vec[:-2] + null*vec[1:-1] + plus*vec[2:]
    return res

def solve_1D_wave_eq(step_params, init_params, bound_params, misc_params):
    """
    Solves the wave equation in 1D using given function and derivative at t = 0.
    The wave is given by a vibrating string of length L starting at x = 0.
    
    step_params: tuple-like
        contains the step sizes for t and x
    init_params: tuple-like
        contains the functions describing initial conditions
    bound_params: tuple-like
        contains the boundary conditions of x
    misc_params: tuple-like
        contains the wave equation parameters
    """

    # step parameters
    N,M,dt = step_params
    # initial condition functions
    init_psi, init_psi_t = init_params
    # boundary values
    psi0,psiN = bound_params
    # wave equation parameters
    c,L = misc_params

    dx = L / N
    psi = np.empty((M+1,N+1))

    # Calculate psi for the first two time points
    psi[0,1:N] = init_psi(np.arange(1,N)*dx)
    psi[:,0] = psi0
    psi[:,N] = psiN

    psi[1,1:N] = psi[0,1:N] + dt*init_psi_t(np.arange(1,N)*dx)

    coefs = c**2 * dt**2 / dx**2 * np.array([1,-2,1])

    shift = (coefs[0]*psi0,coefs[2]*psiN)

    for i in range(2,M+1):
        psi[i,1:N] = 2*psi[i-1,1:N] - psi[i-2,1:N]
        psi[i,1:N] += explicit_tridiagonal(coefs,psi[i-1,1:N],bound_shift=shift)

    return psi

def solve_2D_diffusion_eq(step_params,init_c,bound_params,misc_params):
    """
    Solves the time dependent diffusion equation in 2D
    with periodic boundary conditions in x and dirichlet
    boundary conditions in y

    step_params: tuple-like
        contains the step sizes for t, x and y
    init_c: function pointer
        contains the function describing initial conditions
    bound_params: tuple-like
        contains the boundary condition functions of y
    misc_params: float
        contains the diffusion constant and 
        the minimum and maximum of x and y
    """

    # step parameters
    N,M,dt = step_params
    # boundary condition functions
    bound_y0, bound_yN = bound_params
    # diffusion equation parameters
    D,xmin,xmax = misc_params

    # step size in x and y
    dx = (xmax-xmin) / N

    c = np.empty((M+1,N+1,N+1))

    # initialize the values of c
    c[0,0:N+1,1:N] = init_c(dx*np.arange(0,N+1),dx*np.arange(1,N))

    # insert the boundary conditions
    for k in range(0,M+1):
        c[k,:,0] = bound_y0(dx*np.arange(0,N))
        c[k,:,N] = bound_yN(dx*np.arange(0,N))

    coef = dt * D / dx**2

    for k in range(1,M+1):
        # update normal points
        c[k,1:N,1:N] = (1 - 4* coef) * c[k-1,1:N,1:N]
        c[k,1:N,1:N] += coef * (c[k-1,2:N+1,1:N] + c[k-1,0:N-1,1:N] + c[k-1,1:N,2:N+1] + c[k-1,1:N,0:N-1])

        # update periodic boundaries in x
        c[k,0,1:N] = (1 - 4* coef) * c[k-1,0,1:N]
        c[k,0,1:N] += coef * (c[k-1,1,1:N] + c[k-1,N-1,1:N] + c[k-1,0,2:N+1] + c[k-1,0,0:N-1])
        c[k,N,1:N] = c[k,0,1:N]

    return c

def solve_2D_diffusion_analytic(x,t,N,D):
    i = np.arange(0,N)[:,None]
    term1 = erfc((1-x[None,:]+2*i) / (2*np.sqrt(D*t)))
    term2 = erfc((1+x[None,:]+2*i) / (2*np.sqrt(D*t)))
    return np.sum(term1-term2,axis=0)

def solve_laplace_jacobi(N,eps,init_c,bound_params,misc_params):
    """
    Solves the time independent diffusion equation using Jacobi iteration
    with periodic boundary conditions in x and dirichlet boundary conditions in y
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

    while True:
        # set old values
        c[0,:,:] = c[1,:,:]

        # update normal points
        c[1,1:N,1:N] = 0.25 * (c[0,2:N+1,1:N] + c[0,0:N-1,1:N] + c[0,1:N,2:N+1] + c[0,1:N,0:N-1])

        # update periodic boundaries in x
        c[1,N,1:N] = c[1,0,1:N] = 0.25 * (c[0,1,1:N] + c[0,N-1,1:N] + c[0,0,2:N+1] + c[0,0,0:N-1])

        delta = np.amax(np.abs(c[1,:,:]-c[0,:,:]))
        deltas.append(delta)

        if delta < eps:
            return c[1,:,:],deltas

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

        delta = np.amax(np.abs(c[1,:,:]-c[0,:,:]))
        deltas.append(delta)
        if delta < eps:
            return c[1,:,:],deltas
