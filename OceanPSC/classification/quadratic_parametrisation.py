import numpy as np
from .. import operations as op

def weights(radius):
    """Calculate the inverse distance weights on a (2*radius+1)X(2*radius+1) matrix"""
    n=2*radius+1
    xi=np.arange(0,n)-radius
    yi=np.arange(0,n)-radius
    xi,yi=np.meshgrid(xi,yi)
    d=np.sqrt((xi)**2+(yi)**2)
    w=1/(1+d)
    return w

def coeff_calculator(radius,g):
    """Calculate the 6x6 matrix that will give the coefficients from the surface observation for a given radius
    (the A^-1 in AX=B with X the coefficients and B the observation derived from the surface array)
    - radius is the center-edge distance of the surface array for which the coeffs must be calculated
    - g is the real meter distance between two points"""
    #suppose g=x_unit=y_unit, radius is center/edge dist size=(2r+1)x(2r+1)
    n=2*radius+1
    xi=g*(np.arange(0,n)-radius)
    yi=g*(np.arange(0,n)-radius)
    xi,yi=np.meshgrid(xi,yi)
    wi=weights(radius)
    
    matrix=np.zeros((6,6))
    matrix[0,0]=np.sum(wi**xi**4)
    matrix[0,1]=matrix[1,0]=np.sum(wi*(xi*yi)**2)
    matrix[0,2]=matrix[2,0]=np.sum(wi*yi*xi**3)
    matrix[0,3]=matrix[3,0]=np.sum(wi*xi**3)
    matrix[0,4]=matrix[4,0]=np.sum(wi*yi*xi**2)
    matrix[0,5]=matrix[5,0]=np.sum(wi*xi**2)
    
    matrix[1,1]=np.sum(wi*yi**4)
    matrix[1,2]=matrix[2,1]=np.sum(wi*xi*yi**3)
    matrix[1,3]=matrix[3,1]=np.sum(wi*xi*yi**2)
    matrix[1,4]=matrix[4,1]=np.sum(wi*yi**3)
    matrix[1,5]=matrix[5,1]=np.sum(wi*yi**2)
    
    matrix[2,2]=np.sum(wi*(xi*yi)**2)
    matrix[2,3]=matrix[3,2]=np.sum(wi*yi*xi**2)
    matrix[2,4]=matrix[4,2]=np.sum(wi*xi*yi**2)
    matrix[2,5]=matrix[5,2]=np.sum(wi*xi*yi)
    
    matrix[3,3]=np.sum(wi*xi**2)
    matrix[3,4]=matrix[4,3]=np.sum(wi*xi*yi)
    matrix[3,5]=matrix[5,3]=np.sum(wi*xi)
    
    matrix[4,4]=np.sum(wi*yi**2)
    matrix[4,5]=matrix[5,4]=np.sum(wi*yi)
    matrix[5,5]=np.sum(wi)
    return np.linalg.inv(matrix)

def B_vector(surface,g):#nxn numpy array (or (*)xnxn)
    """The observation derived from the surface to be used in the coefficient calculation
    (the B in AX=B with A the coeff calculator for the given radius and X the coefficients)
    -surface must be an nxn or (*)xnxn np array
    -g is the real meter distance between two points"""
    n=surface.shape[-1]
    radius=(n-1)/2
    xi=g*(np.arange(n)-radius)
    yi=g*(np.arange(n)-radius)
    xi,yi=np.meshgrid(xi,yi)
    wi=weights(radius)
    results=np.zeros([6]+list(surface.shape[:-2]))
    results[0]=(wi*surface*xi**2).sum(axis=(-1,-2))
    results[1]=(wi*surface*yi**2).sum(axis=(-1,-2))
    results[2]=(wi*surface*xi*yi).sum(axis=(-1,-2))
    results[3]=(wi*surface*xi).sum(axis=(-1,-2))
    results[4]=(wi*surface*yi).sum(axis=(-1,-2))
    results[5]=(wi*surface).sum(axis=(-1,-2))
    return results.transpose(1,2,0)


def quadratic_parametrisation(study_area,radius,g):
    """Calculate the 6 surface caracterisation coefficients at each point for the entire study area.
    Each surface will be locally fitted to the following quadratic function (with origin at the center):
    f(x,y)=ax^2 + by^2 +cxy + dx + ey + f.
    The coeffs are calculated by way of solving a linear system on each surface.
    entry: study_area: (N,M) 2D shape
    out: a,b,c,d,e,f: 6 values, each in the same (N,M) 2D shape"""
    
    #the goal is to solve the AX=B equation to get X=(a,b,c,d,e,f) the 6 coefficients
    #first, let's calculate the surface around each point, surface to which we must fit the quadratic estimation
    vois=op.neighbor_grid(study_area,radius,bound_method='duplicate')
    #step2, calculate B, the observation vector and the second hand of the AX=B equation
    B=B_vector(vois,g)
    #then let's calculate the inverse of A, since AX=B <==> X=(A^-1)B 
    Ainv=coeff_calculator(radius,g)

    #now, we just have to calculate X=(A^-1)B
    #B is reshaped to a Mx6 matrix which will be multipliable by a 6x6 matrix
    longB=B.reshape(-1,6)
    #if longB is a Mx6 matrix, and A.T is a 6x6 matrix, with longB=(B1,...,BM)
    #then longB@(Ainv.T)=(Ainv@B1,...,Ainv@BM) which is our desired result
    #Note: A is symetric so Ainv should be too: but it is not, because of numpy inverse calculation errors
    #all there is left to do, is to reshape the results in the original study area (N,M) shape, so each result is on the right point
    coeffs=(longB@(Ainv.T)).reshape(list(B.shape[:2])+[6])
    #then, we can just retrieve the coefficients
    a,b,c,d,e,f=coeffs.transpose(2,0,1)
    return a,b,c,d,e,f
