"""
Author: Kevin de Vries
Student number: 10579869

Produces an animation of the integration of the 
time dependent diffusion equation over time.

Animation template taken from: 
https://matplotlib.org/examples/animation/simple_anim.html
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from source_exercise1 import *

# Define initial condition
init_c = lambda x,y: 0

# Define boundary conditions
bound_y0 = lambda x: 0
bound_yN = lambda x: 1

# Initialize the fixed parameters
step = N,M,dt = 50,10000,0.0001
misc = D,xmin,xmax = 1,0,1

# Solve the diffusion equation
c = solve_2D_diffusion_eq(step,init_c,(bound_y0,bound_yN),misc)

fig = plt.figure(figsize=(20,10))

im = plt.imshow(c[0,:,:],origin='lower',cmap="jet",animated=True)

plt.title("Time integration of the time dependent diffusion equation")

plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

plt.xticks(np.arange(0,N+1,5),(xmax-xmin) * np.arange(0,N+1,5) / N)
plt.yticks(np.arange(0,N+1,5),(xmax-xmin) * np.arange(0,N+1,5) / N)

plt.colorbar(im)

def animate(i):
    im.set_array(c[i,:,:].T)
    return im,

ani = animation.FuncAnimation(fig, animate, np.arange(1, M+1),
                              interval=0, blit=True)
plt.show()
