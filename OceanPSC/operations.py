import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.lib import stride_tricks
import skimage.measure

def neighbor_grid(grid,neighbor_dist=1, round_earth=False):
    """Turns a n x m grid into a n x m x (2*neighbor_dist +1)x(2*neighbor_dist +1) by replacing each point by a grid containing the points and its neighbors"""
    n,m=grid.shape
    #growing the grid to give neighbors to border points
    rgrid=np.zeros((n+2*neighbor_dist,m+2*neighbor_dist))
    #the base grid is set at the center of the new grid
    rgrid[neighbor_dist:-neighbor_dist,neighbor_dist:-neighbor_dist]=grid
    #border filled by assuming round_earth
    if round_earth:
        rgrid[0:neighbor_dist,neighbor_dist:-neighbor_dist]=grid[-neighbor_dist-1:-1,:]
        rgrid[-neighbor_dist-1:-1,neighbor_dist:-neighbor_dist]=grid[0:neighbor_dist,:]
        rgrid[neighbor_dist:-neighbor_dist,0:neighbor_dist]=grid[:,-1-neighbor_dist:-1]
        rgrid[neighbor_dist:-neighbor_dist,-neighbor_dist-1:-1]=grid[:,0:neighbor_dist]
    #striding
    #the n,m array is changed into a n,m,3,3 array, each cell is replaced by a 3x3 grid containing the cell
    #and its 8 neighbours, not 3 or 8 if neighbor_dist is changed
    shape = (n, m, (2*neighbor_dist+1), (2*neighbor_dist+1))
    patches = stride_tricks.as_strided(rgrid, shape=shape, strides=(2* rgrid.strides))
    return patches

def laplacien(na,x_dist=1,y_dist=1):
    """Calcule le champ du laplacien à partir du champ des valeurs."""
    vois=neighbor_grid(na)
    filtre=(1/4)*np.array([[0,1/y_dist,0],
                      [1/x_dist,-2/x_dist -2/y_dist,1/x_dist],
                      [0,1/y_dist,0]])
    laplacien=np.sum(filtre * vois, axis=(-1,-2))
    return clean_boundaries(laplacien)

def grad(na,x_dist=1, y_dist=1,clean_bounds=True):
    """Calcule le champ vectoriel du gradient à partir du champ des valeurs"""
    vois=neighbor_grid(na)
    filtre1=(1/2*x_dist)*np.array([[0,0,0],
                      [-1,0,1],
                      [0,0,0]])
    filtre2=(1/2*y_dist)*np.array([[0,-1,0],
                                    [0,0,0],
                                    [0,1,0]])
    gradX=np.sum(filtre1 * vois, axis=(-1,-2))
    gradY=np.sum(filtre2 * vois, axis=(-1,-2))
    if clean_bounds:
        gradX=clean_boundaries(gradX)
        gradY=clean_boundaries(gradY)
    rsl=np.array([gradX,gradY]).T
    return rsl

def clean_boundaries(na, length=1):
    na[:,-length-1:-1]=0
    na[0:length,:]=0
    na[:,0:length]=0
    na[-length-1:-1,:]=0
    return na

def norme(na):
    """Norme d'un champs de vecteurs (n x m x 2)."""
    X,Y=na.T
    return np.sqrt(X**2 + Y**2)

def normalize(grid):
    b=np.min(grid)
    grid=grid-np.min(grid)
    a=np.max(grid)
    grid=grid/a
    return grid,a,b

def create_indicator_grid(data,calc,reduction=(54,54)):
    red_mean = skimage.measure.block_reduce(data, reduction, calc)
    red_mean,red_mean_a,red_mean_b=normalize(red_mean)
    return red_mean,red_mean_a,red_mean_b