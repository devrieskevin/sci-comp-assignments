"""
Author: Kevin de Vries
Student number: 10579869

Produces an animation of the integration of the 
Gray-Scott reaction diffusion model over time using
periodic boundary conditions in both the x and y-direction.

Animation template taken from: 
https://matplotlib.org/examples/animation/simple_anim.html
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

from source_exercise2 import *

# Initialize the fixed parameters
step = N,M,dx,dt = 100,10000,1.0,1.0
misc = f,k,Du,Dv = np.array(sys.argv[1:],dtype=float)

def init_u(x,y,scale=0.01):
    return 0.5 + scale * np.random.uniform(-1,1,(x.size,y.size))

def init_v(x,y,scale=0.01):
    res = scale * np.random.uniform(0,1,(x.size,y.size))
    res[x.size//2-2:x.size//2+3,y.size//2-2:y.size//2+3] = 0.25
    #res[x.size//2-2-20:x.size//2+3-20,y.size//2-2-20:y.size//2+3-20] = 0.25
    return res

# Solve the rection-diffusion equation
u,v = gray_scott_periodic(step, (init_u,init_v), misc)

fig,(ax_u,ax_v) = plt.subplots(1,2,figsize=(10,20))

im_u = ax_u.imshow(u[0,:,:],extent=[0,dx*N,0,dx*N],origin='lower',cmap="jet",animated=True)
im_v = ax_v.imshow(v[0,:,:],extent=[0,dx*N,0,dx*N],origin='lower',cmap="jet",animated=True)

# Colorbars do not seem to be able to animate
#cb_u = fig.colorbar(im_u,ax=ax_u)
#cb_v = fig.colorbar(im_v,ax=ax_v)

ax_u.set_title(r"$u$")
ax_v.set_title(r"$v$")

def animate(i):
    im_u.set_array(u[i,:,:].T)
    im_v.set_array(v[i,:,:].T)

    umax = np.max(u[i,:,:])
    umin = np.min(u[i,:,:])

    vmax = np.max(v[i,:,:])
    vmin = np.min(v[i,:,:])

    im_u.set_clim(umin,umax)
    im_v.set_clim(vmin,vmax)

    return im_u,im_v

ani = animation.FuncAnimation(fig, animate, np.arange(1, M+1),
                              interval=0, blit=True)
plt.show()
