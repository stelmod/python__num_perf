
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import colors


# In[2]:

def mset_draw(mset):
    plt.imshow(mset, norm=colors.PowerNorm(0.3), cmap='cubehelix');


# In[3]:

def create_intervals(xmin, xmax, ymin, ymax, width, height):
    return np.linspace(xmin, xmax, width), np.linspace(ymin, ymax, height)


# In[4]:

def mset_iteration(c, maxiter=256):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z**2 + c
    return n


# In[5]:

def mandelbrot_set_list_comp(xmin, xmax, ymin, ymax, width, height, maxiter=256):
    real_range, imaginary_range = create_intervals(xmin, xmax, ymin, ymax, width, height)

    m = [mset_iteration(r + s*1j, maxiter) for r in real_range for s in imaginary_range]
    return m, real_range, imaginary_range


# In[6]:

mset, r, i = mandelbrot_set_list_comp(-2.0,0.5,-1.25,1.25, 600, 600)
mset_draw(np.array(mset).reshape(600, 600).T);


# In[7]:

get_ipython().magic('timeit mandelbrot_set_list_comp(-2.0,0.5,-1.25,1.25, 600, 600)')


# In[9]:

def mandelbrot_set_numpy(xmin, xmax, ymin, ymax, width, height, maxiter=256):
    m = np.empty((height, width), dtype=np.uint8)
    real_range, imaginary_range = create_intervals(xmin, xmax, ymin, ymax, width, height)
    
    for j, i in product(range(height), range(width)):
        x = real_range[i]
        y = imaginary_range[j]
        c = x + y*1j
        m[j,i] = mset_iteration(c, maxiter)

    return m, real_range, imaginary_range


# In[12]:

#mset_draw(mandelbrot_set_numpy(-2.0,0.5,-1.25,1.25, 600, 600)[0])


# In[13]:

get_ipython().magic('timeit mandelbrot_set_numpy(-2.0,0.5,-1.25,1.25, 600, 600)')


# In[16]:

def mandelbrot_vectors(c, maxiter):
    output = np.zeros(c.shape)
    z = np.zeros(c.shape, np.complex64)
    for it in range(maxiter):
        notdone = np.less(z.real*z.real + z.imag*z.imag, 4.0)
        output[notdone] = it
        z[notdone] = z[notdone]**2 + c[notdone]
    output[output == maxiter-1] = maxiter-1
    return output

def mandelbrot_set_vectors(xmin,xmax,ymin,ymax,width,height,maxiter=256):
    real_range, imaginary_range = create_intervals(xmin, xmax, ymin, ymax, width, height)

    c = real_range + imaginary_range[:,None]*1j
    return mandelbrot_vectors(c, maxiter)


# In[18]:

#mset_draw(mandelbrot_set_vectors(-2.0,0.5,-1.25,1.25, 600, 600))


# In[20]:

get_ipython().magic('timeit mandelbrot_set_vectors(-2.0,0.5,-1.25,1.25, 600, 600)')


# In[21]:

from numba import jit

@jit
def mset_iteration_numba(c, maxiter=256):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z**2 + c
    return n

@jit
def mandelbrot_set_numba(xmin, xmax, ymin, ymax, width, height, maxiter=256):
    m = np.empty((height, width), dtype=np.uint8)
    real_range, imaginary_range = create_intervals(xmin, xmax, ymin, ymax, width, height)

    for j, i in product(range(height), range(width)):
        x = real_range[i]
        y = imaginary_range[j]
        c = x + y*1j
        m[j,i] = mset_iteration_numba(c, maxiter)

    return m, real_range, imaginary_range


# In[23]:

#mset_draw(mandelbrot_set_numba(-2.0,0.5,-1.25,1.25, 600, 600)[0])


# In[24]:

get_ipython().magic('timeit mandelbrot_set_numba(-2.0,0.5,-1.25,1.25, 600, 600)')


# In[ ]:



