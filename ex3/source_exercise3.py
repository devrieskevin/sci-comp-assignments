"""
Author:         Kevin de Vries
Student number: 10579869

"""

import numpy as np

import scipy.linalg as linalg
import scipy.sparse.linalg as sp_linalg
import scipy.sparse as sp

from scipy.sparse import diags


def matrix_drum_rectangular(N,misc):
    """
    Constructs the matrix for the eigenvalue problem 
    resulting from the discretization of the spacial component
    of the wave equation in 2D.
    
    This is implemented for any rectangular drum or membrane.
    The boundaries of the rectangle are thus set to 0, meaning
    that the drum or membrane is fixed at the boundaries.
    """
    
    Lx,Ly = misc
    
    # Calculate step size in the x direction
    dx = Lx / N
    
    # Calculate number of steps in the y-direction (rectangle)
    M = int(round(Ly / dx))
    
    # Dimension of the matrix (spacial dimensions flattened)
    dim = (N-1)*(M-1)
    
    offdiag_1 = np.ones(dim)
    offdiag_1[M-2::M-1] = 0
    
    diagonals = [1,offdiag_1,-4,offdiag_1,1]
    positions = [-M+1,-1,0,1,M-1]
    
    # Construct resulting matrix
    res = diags(diagonals,positions,shape=(dim,dim))
    
    # Apply scale factor to the matrix
    res /= dx**2
    
    return (N,M),res

def matrix_drum_circular(N,L):
    """
    Constructs the matrix for the eigenvalue problem 
    resulting from the discretization of the spacial component
    of the wave equation in 2D.
    
    This is implemented for a circular drum or membrane.
    The boundaries of the circle are thus set to 0, meaning
    that the drum or membrane is fixed at the boundaries.
    """
    
    # Calculate step size
    dx = L / N
    
    # Calculate the radius of the circle
    R = L / 2
    
    # Calculate the distance to the origin (radius) per grid point
    idx_matrix = np.tile(np.arange(N+1),(N+1,1))
    distances = np.sqrt((idx_matrix * dx - R)**2 + (idx_matrix.T * dx - R)**2)
    
    # Determine which grid points to include
    inclusion = distances < R
    
    # Construct index vector
    idx_vec = np.arange(inclusion.size)[inclusion.flatten()]
    
    # Initialize the matrix as sparse matrix with given diagonals
    res = diags([-4],[0],shape=(idx_vec.size,idx_vec.size))
    
    # Convert to LIL matrix for matrix construction
    res = sp.lil_matrix(res)
    
    # Construct matrix using index mapping
    for n in range(idx_vec.size):
        idx = idx_vec[n]
        
        if n > 0 and idx_vec[n-1] == idx - 1:
            res[n,n-1] = 1
        if n < idx_vec.size-1 and idx_vec[n+1] == idx + 1:
            res[n,n+1] = 1
        
        res[n,idx_vec == (idx - N - 1)] = 1
        res[n,idx_vec == (idx + N + 1)] = 1
    
    # Apply scale factor to the matrix
    res /= dx**2
    
    # Convert back to CSR matrix for faster arithmetic
    res = res.tocsr()
    
    return res,inclusion

def diffusion_equation_direct(N,L,sources):
    """
    Calculates numerical solution of the laplace equation
    using a direct method, which solves a system of linear equations.
    
    The boundary used is a circular disk on and outside of which c = 0.
    Sources of concentration can be added to the disk.
    """
    
    # Calculate step size
    dx = L / N
    
    # Calculate the radius of the circle
    R = L / 2
    
    # Get source coordinates and concentrations
    sx,sy,sc = sources
    
    # Offset source coordinates with radius
    sx += R
    sy += R
    
    # Calculate the distance to the origin (radius) per grid point
    idx_matrix = np.tile(np.arange(N+1),(N+1,1))
    distances = np.sqrt((idx_matrix * dx - R)**2 + (idx_matrix.T * dx - R)**2)
    
    # Map concentration source positions to indices
    source_map = np.zeros(idx_matrix.size,dtype=bool)
    sc_map = np.empty(idx_matrix.size)
    for n in range(sc.size):
        source_dist = np.sqrt((idx_matrix * dx - sy[n])**2 + (idx_matrix.T * dx - sx[n])**2)
        idx = np.argmin(source_dist)
        source_map[idx] = True
        sc_map[idx] = sc[n]
        
    # Retrieve ordered source concentrations
    sc_ordered = sc_map[source_map]
    
    # Determine which grid points to include
    inclusion = distances < R
    inclusion = (inclusion.flatten() & ~source_map).reshape(N+1,N+1)
    
    # Construct index vector
    idx_vec = np.arange(inclusion.size)[inclusion.flatten()]
    
    b = np.zeros(idx_vec.size)
    
    # Calculate values of b elements from the sources
    source_idx = np.arange(source_map.size)[source_map]
    for n in range(source_idx.size):
        idx = source_idx[n]
        cur_sc = sc_ordered[n]
        
        b[idx_vec == (idx-1)] -= cur_sc
        b[idx_vec == (idx+1)] -= cur_sc
        b[idx_vec == (idx - N - 1)] -= cur_sc
        b[idx_vec == (idx + N + 1)] -= cur_sc
    
    # Initialize matrix
    M = diags([-4],[0],shape=(idx_vec.size,idx_vec.size))
    M = sp.lil_matrix(M)
    
    # Construct matrix using index mapping
    for n in range(idx_vec.size):
        idx = idx_vec[n]
        
        if n > 0 and idx_vec[n-1] == idx - 1:
            M[n,n-1] = 1
        if n < idx_vec.size-1 and idx_vec[n+1] == idx + 1:
            M[n,n+1] = 1
        
        M[n,idx_vec == (idx - N - 1)] = 1
        M[n,idx_vec == (idx + N + 1)] = 1
    
    # Solve linear system of equations to get the concentrations
    conc = sp_linalg.spsolve(M.tocsr(),b)
    
    return conc,inclusion,source_map.reshape(N+1,N+1),sc_ordered


















