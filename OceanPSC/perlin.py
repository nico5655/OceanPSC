from argparse import ZERO_OR_MORE
from zipfile import ZIP64_LIMIT
import numpy as np
import matplotlib.pyplot as plt
from time import time

#np.random.seed(1002)
Vec = np.random.rand(256,256,2)
V = 2 * np.random.rand(256,256) - 1
#Px = np.random.permutation(256)
#Py = np.random.permutation(256)
Vec = (Vec.T / np.sqrt(np.sum(Vec ** 2,axis=-1)).T).T


def Smooth(t):
    return t * t * t * (t * (t * 6 - 15) + 10)
    #return 3*t*t-t*t*t
def Val(x,y): #valeurs aux neouds
    return V[x % 256,y % 256]

def Vecteur(x,y):
    a = Vec[x % 256,y % 256]
    return a


def Bruit(x,y,vecteur=Vecteur,val=Val):
    if x is float:
        x=np.array([x])
        y=np.array([y])

    X,Y = np.int32(x),np.int32(y)
    x,y = x - X,y - Y
    sx,sy = Smooth(x), Smooth(y)

    bruit = 0
    bruit += (1 - sx) * (1 - sy) * (val(X,Y) + np.sum(vecteur(X,Y) * np.array([x.T,y.T]).T,axis=-1))
    bruit += sx * (1 - sy) * (val(X + 1,Y) + np.sum(vecteur(X + 1,Y) * np.array([x.T - 1,y.T]).T,axis=-1))
    bruit += (1 - sx) * sy * (val(X,Y + 1) + np.sum(vecteur(X,Y + 1) * np.array([x.T,y.T - 1]).T,axis=-1))
    bruit += sx * sy * (val(X + 1,Y + 1) + np.sum(vecteur(X + 1,Y + 1) * np.array([x.T - 1,y.T - 1]).T,axis=-1))

    return bruit

def perlin(x,y,alpha,omega,n=10):
    N = 0
    a = 1
    z = x + y * 1j
    for k in range(n):
        N += a * Bruit(z.real,z.imag)
        z *= omega
        a *= alpha
    N=N-np.min(N)
    N=N/np.max(N)
    return N/2-1

def afficher(n):
    t_1 = time()
    alpha = 1 / 2
    omega = 2
    i = np.arange(n)
    j = np.arange(n)
    I,J = np.meshgrid(i,j)
    carte = perlin(J / 50,I / 50,alpha,omega,8)
    print(time() - t_1)
    """t_1 = time()
    carte=np.array([[perlin(i/50,j/50,alpha,omega,8) for i in range(n)] for j in range(n)])
    print(time() - t_1)"""
    output_path = "sim-final4.png"
    plt.imsave(output_path,carte,cmap='gray')
    #show3d(carte)

if __name__ == 'main':
    afficher(200)

#C'est lent : on peut surement acc�l�rer en faisant les op�rations sur tout le tableau avec numpy, optimiser les param�tres et le code 
#Plusieurs pistes d'am�lioration : le choix des vecteurs al�atoire, on peut faire autre chose que %256 pour que ca ne boucle pas
#-un bruit 3D projet� comme dans l'article Better Gradient Noise, Andrew Kensler, Aaron Knoll and Peter Shirley,UUSCI-2008-001
#-Comme dans l'article State of the Art in Procedural Noise Functions, on peut orienter le bruit, et le modifier de plusieurs mani�res.
