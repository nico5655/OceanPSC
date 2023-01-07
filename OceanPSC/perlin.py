from argparse import ZERO_OR_MORE
from zipfile import ZIP64_LIMIT
import numpy as np

n=100
np.random.seed(1002)
Vecteur = np.random.rand(256,256,2)
Px = np.random.permutation(256)
Py = np.random.permutation(256)
for i in range(256):
    for j in range(256):
        Vecteur[i][j] /= np.sqrt(np.sum(Vecteur[i][j]**2)) #normalisation

def Smooth(t):
    return t * t * t * (t * (t * 6 - 15) + 10)
    #return 3*t*t-t*t*t

def Hash(x,y): #valeurs aux neouds
    return 0
def Bruit(x,y):
    X,Y = int(x),int(y)
    x,y = x - X,y - Y
    u,v = Smooth(x), Smooth(y)
    bruit = Hash(x,y)
    for i in range(2):
        for j in range(2):
            bruit += (i-u)*(j-v)*np.dot(Vecteur[Px[(X+i)%256],Py[(Y+j)%256]],[x,y])*(1-2*((i-j)!=0)) #on ajoute 4 vecteurs avec des poids 
    return bruit

def perlin(x,y,alpha,omega,n=10):
    N = 0
    for k in range(n):
        z = omega**n*(x+y*1j)
        N += alpha**k*Bruit(z.real,z.imag)
    return N

