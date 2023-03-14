import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.lib import stride_tricks
import skimage.measure

def neighbor_grid(grid,neighbor_dist=1, bound_method='empty'):#empty, duplicate
    """Turns a n x m grid into a n x m x (2*neighbor_dist +1)x(2*neighbor_dist +1) by replacing each point by a grid containing the points and its neighbors"""
    n,m=grid.shape
    #growing the grid to give neighbors to border points
    #the base grid is set at the center of the new grid
    if bound_method == 'duplicate':
        rgrid=np.pad(grid,neighbor_dist,mode='symmetric')
    else:
        rgrid=np.pad(grid,neighbor_dist,mode='constant',constant_values=0)

    #striding
    #the n,m array is changed into a n,m,3,3 array, each cell is replaced by a 3x3 grid containing the cell
    #and its 8 neighbours, not 3 or 8 if neighbor_dist is changed
    shape = (n, m, (2*neighbor_dist+1), (2*neighbor_dist+1))
    patches = stride_tricks.as_strided(rgrid, shape=shape, strides=(2* rgrid.strides))
    return patches

def divide_treatment(data,f,vertical_cut=False,r=5,result_len=1,print_progress=False):
    border=50#cut on vertical,circumnavigate on horizontal
    n,m=data.shape
    if not vertical_cut:
        n_data=np.zeros((n+2*border,m))
        n_data[border:-border,:]=data
        n_data[:border,:]=data[:border,:][::-1,:]
        n_data[-border:,:]=data[-border:,:][::-1,:]
        data=n_data
    else:
        n-=2*border
    
    sn=n//r
    sm=m//r
    w_data=np.zeros((n+2*border,m+2*border))
    w_data[:,border:-border]=data
    w_data[:,:border]=data[:,-border:]
    w_data[:,-border:]=data[:,:border]
    if result_len==1:
        result=np.zeros((n,m))
    else:
        result=np.zeros((n,m,result_len))
    for i in range(r):
        for j in range(r):
            area=w_data[(i*sn):((i+1)*sn+2*border),(j*sm):((j+1)*sm+2*border)]
            if result_len==1:
                result[(i*sn):((i+1)*sn),(j*sm):((j+1)*sm)]=f(area)[border:-border,border:-border]
            else:
                result[(i*sn):((i+1)*sn),(j*sm):((j+1)*sm)]=np.array(f(area)).transpose(1,2,0)[border:-border,border:-border]
            if print_progress:
                print(100*(i*r+j+1)/(r**2),'% done')
    if result_len==1:
        return result
    return result.transpose(2,0,1)


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
    filtre1=(1/(8*x_dist))*np.array([[-1,0,1],
                                    [-2,0,2],
                                    [-1,0,1]])

    filtre2=(1/(8*y_dist))*np.array([[-1,-2,-1],
                                    [0,0,0],
                                    [1,2,1]])

    gradX=np.sum(filtre1 * vois, axis=(-1,-2))
    gradY=np.sum(filtre2 * vois, axis=(-1,-2))
    if clean_bounds:
        gradX=clean_boundaries(gradX)
        gradY=clean_boundaries(gradY)
    rsl=np.array([gradX,gradY]).T#todo: regarder les transposée... (et checker les dimensions)
    return rsl


def clean_boundaries(na, length=1):
    na[:,-length-1:-1]=0
    na[0:length,:]=0
    na[:,0:length]=0
    na[-length-1:-1,:]=0
    return na

def clean_boundaries_dupl(na,length=1):
    na[:,-length-1:-1]=na[:,-2*length-1:-length-1]
    na[0:length,:]=na[length:2*length,:]
    na[:,0:length]=na[:,length:2*length]
    na[-length-1:-1,:]=na[-2*length-1:-length-1,:]
    return na

def norme(na):
    """Norme d'un champs de vecteurs (n x m x 2)."""
    X,Y=na.T
    return np.sqrt(X**2 + Y**2)

def normalize(grid):
    if np.max(grid)==np.min(grid):
        return grid/np.max(grid),np.min(grid),np.max(grid)
    b=np.min(grid)
    grid=grid-np.min(grid)
    a=np.max(grid)
    grid=grid/a
    return grid,a,b

def create_indicator_grid(data,calc,reduction=(54,54),normaliser=True):
    red_mean = skimage.measure.block_reduce(data, reduction, calc)
    if normaliser:
        red_mean,red_mean_a,red_mean_b=normalize(red_mean)
    return red_mean,1,0