"""
Author: Kevin de Vries
Student number: 10579869

Produces an animation of the wave equation integrated over time.

Animation template taken from: 
https://matplotlib.org/examples/animation/simple_anim.html
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from source_exercise1 import *

# Define the initial condition functions
init_psi_1 = lambda x: np.sin(2*np.pi*x)
init_psi_2 = lambda x: np.sin(5*np.pi*x)

def init_psi_3(x):
    res = init_psi_2(x)
    res[x < 1/5] = 0
    res[x > 2/5] = 0
    return res

# Define the initial conditions
init_psi_t = lambda x: 0

# Initialize the fixed parameters
step = N,M,dt = 1000,2000,0.001
bound = psi0,psiN = 0,0
misc = c,L = 1,1

# Solve the wave equations with different initial conditions
psi1 = solve_1D_wave_eq(step,(init_psi_1,init_psi_t),bound,misc)
psi2 = solve_1D_wave_eq(step,(init_psi_2,init_psi_t),bound,misc)
psi3 = solve_1D_wave_eq(step,(init_psi_3,init_psi_t),bound,misc)

# Initialize the figure
fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,10),sharey=True)

# Generate the discretized x values
x = L/N * np.arange(0,N+1)

line1, = ax1.plot(x, psi1[0,:])
line2, = ax2.plot(x, psi2[0,:])
line3, = ax3.plot(x, psi3[0,:])

ax1.set_title(r"$\Psi(x,t=0) = \sin(2 \pi x)$")
ax2.set_title(r"$\Psi(x,t=0) = \sin(5 \pi x)$")
ax3.set_title(r"$\Psi(x,t=0) = \sin(5 \pi x)$" + "\n" + 
              r"if $\frac{1}{5}<x<\frac{2}{5}$ else $\Psi(x,t=0) = 0$")

ax1.set_xlabel(r"$x$")
ax2.set_xlabel(r"$x$")
ax3.set_xlabel(r"$x$")

ax1.set_ylabel(r"$\Psi$")

ax1.grid(True)
ax2.grid(True)
ax3.grid(True)

def animate(i):
    line1.set_ydata(psi1[i,:])
    line2.set_ydata(psi2[i,:])
    line3.set_ydata(psi3[i,:])
    return line1,line2,line3


# Init only required for blitting to give a clean slate.
def init():
    line1.set_ydata(np.ma.array(x, mask=True))
    line2.set_ydata(np.ma.array(x, mask=True))
    line3.set_ydata(np.ma.array(x, mask=True))
    return line1,line2,line3

ani = animation.FuncAnimation(fig, animate, np.arange(1, M+1), init_func=init,
                              interval=0, blit=True)
plt.show()
