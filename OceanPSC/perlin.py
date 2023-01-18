from argparse import ZERO_OR_MORE
from zipfile import ZIP64_LIMIT
import numpy as np
import matplotlib.pyplot as plt
from time import time

np.random.seed(1001)
Vec = np.random.rand(256,256,2)
V = 2*np.random.rand(256,256)-1
#Px = np.random.permutation(256)
#Py = np.random.permutation(256)
for i in range(256):
    for j in range(256):
        Vec[i][j] /= np.sqrt(np.sum(Vec[i][j]**2)) #normalisation

def Smooth(t):
    return t * t * t * (t * (t * 6 - 15) + 10)
    #return 3*t*t-t*t*t

def Val(x,y): #valeurs aux neouds
    return V[x%256][y%256]

def Vecteur(x,y):
    return Vec[x%256][y%256]

def Bruit(x,y):
    X,Y = int(x),int(y)
    x,y = x - X,y - Y
    sx,sy = Smooth(x), Smooth(y)
    bruit = 0
    bruit += (1-sx)*(1-sy)*(Val(X,Y)+np.dot(Vecteur(X,Y),[x,y]))
    bruit += sx*(1-sy)*(Val(X+1,Y)+np.dot(Vecteur(X+1,Y),[x-1,y]))
    bruit += (1-sx)*sy*(Val(X,Y+1)+np.dot(Vecteur(X,Y+1),[x,y-1]))
    bruit += sx*sy*(Val(X+1,Y+1)+np.dot(Vecteur(X+1,Y+1),[x-1,y-1]))

    return bruit

def perlin(x,y,alpha,omega,n=10):
    N = 0
    a = 1
    z = x+y*1j
    for k in range(n):
        N += a*Bruit(z.real,z.imag)
        z *= omega
        a *= alpha
    return N

def afficher(n):
    t_1 = time()
    alpha = 1/2
    omega = 2
    carte = [[perlin(i/50,j/50,alpha,omega,8) for i in range(n)] for j in range(n)]
    print(time()-t_1)
    plt.imshow(carte, cmap='terrain')
    plt.show()
    #show3d(carte)

afficher(200)

#C'est lent : on peut surement acc�l�rer en faisant les op�rations sur tout le tableau avec numpy, optimiser les param�tres et le code 
#Plusieurs pistes d'am�lioration : le choix des vecteurs al�atoire, on peut faire autre chose que %256 pour que ca ne boucle pas
#-un bruit 3D projet� comme dans l'article Better Gradient Noise, Andrew Kensler, Aaron Knoll and Peter Shirley,UUSCI-2008-001
#-Comme dans l'article State of the Art in Procedural Noise Functions, on peut orienter le bruit, et le modifier de plusieurs mani�res.
