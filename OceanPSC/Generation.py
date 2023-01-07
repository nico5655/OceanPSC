if __name__ == "__main__":
    import sys
    from pathlib import Path

    file = Path(__file__).resolve()
    parent, root = file.parent, file.parents[1]
    sys.path.append(str(root))

from re import T
import time
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from OceanPSC.Map import Map
from mpl_toolkits import mplot3d
from matplotlib import cm
import utils
from matplotlib.colors import LightSource
from Erosion import erosion
from perlin import perlin

#On prend une carte avec les continents, on genere les fonds en
#1-Donnant un type a chaque tuile
#2-Generant les fronti�res
#3-Generant les tuiles
resolution = 7
taille_tuile = 2 ** resolution
hauteur = 2
largeur = 2
Carte = Map(-1000 * np.ones((largeur * taille_tuile,hauteur * taille_tuile)))
Liste_types = [(-1000,10,200),(-850,20,100),(-850,20,400),(-1100,20,200)] #Listes de tous les types, et de ses caract�ristiques alt moy, grad moy,
                                                                          #amplitude min/max
def trouver_Ocean(Carte):
    # retourne les tuiles sur lesquelles il faut generer
    return Carte.indicator_grid(np.max, lambda x: np.float32(x.data <= 0), normaliser=False)


def attribution_types(Carte, Types):
    Tab_types = [
        [0] * hauteur for _ in range(largeur)
    ]  # il faut des int pas les float de numpy
    return Tab_types



def genere_alt_moyennes(Carte,Tab_types):
    n,m = Tab_types.shape
    alt_moy = np.zeros((n,m))
    alt_moy_etendue = np.zeros((n + 1,m + 1))
    for k in range((len(Liste_types))):
        alt_moy[Tab_types == k] = Liste_types[k][0]
    alt_moy_etendue[:-1,:-1] = alt_moy
    alt_moy_etendue[-1,:] = alt_moy_etendue[-2,:]
    alt_moy_etendue[:,-1] = alt_moy_etendue[:,-2]

    def S(x):
        return 3 * x * x - 2 * x * x * x


    #Pseudo-random numbers, entre -1,1
    def a(i,j):
        return alt_moy_etendue[i,j]

    #Noise local : donne la valeur en (x,y) de la fonction lisse qui relie les
    #4 points (i,j) ,...,(i+1,j+1)
    def N(x,y):

        i,j = np.int32(np.trunc(x)),np.int32(np.trunc(y))
        return a(i,j) + (a(i + 1,j) - a(i,j)) * S(x - i) + (a(i,j + 1) - a(i,j)) * S(y - j) + (a(i,j) - a(i,j + 1) - a(i + 1,j) + a(i + 1,j + 1)) * S(x - i) * S(y - j)

    I = np.arange(0,n * taille_tuile)
    J = np.arange(0,m * taille_tuile)
    I,J = np.meshgrid(I,J)
    Carte.data = N(I / taille_tuile,J / taille_tuile)
    plt.imshow(Carte.data,cmap='terrain')
    plt.show()


def genere_tuile(Carte,type_tuile,i_tuile,j_tuile):
    
    altitude_moyenne = type_tuile[0]
    gradient_moyen = type_tuile[1]
    amplitude_altitude = type_tuile[2]

    #I = np.arange(0,taille_tuile)
    #J = np.arange(0,taille_tuile)
    #I,J = np.meshgrid(I,J)

    #i_carte = i_tuile + I
    #j_carte = j_tuile + J
    alpha=1/2
    omega=2
    for i in range(taille_tuile):
        for j in range(taille_tuile):
            i_carte,j_carte = i_tuile+i,j_tuile+j
            Carte.data[i_carte,j_carte] +=  amplitude_altitude * perlin(i_carte,j_carte,alpha,omega) #+bonne moyenne continue

#le résultat n'est pas incroyable, peut etre il faut garder les ^mêmes nombre
#aléatoire et simplement en rajouter ?
def show3d(map2):
    x = np.linspace(0, hauteur * taille_tuile, hauteur * taille_tuile)
    y = np.linspace(0, largeur * taille_tuile, largeur * taille_tuile)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(map2, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(X,Y, map2, rstride=4, cstride=4, facecolors=rgb,
                           linewidth=0, antialiased=False, shade=True)
    fig.colorbar(surf)
    plt.show()


if __name__ == "__main__":
    Tuile_a_generer = trouver_Ocean(Carte)
    #Tab_types = attribution_types(Carte, Liste_types)

    genere_tuile(Carte, Liste_types[0], 0, 0)
    Tab_types = np.int32(np.trunc(np.random.rand(hauteur,largeur) * len(Liste_types)))
    genere_alt_moyennes(Carte,Tab_types)

    for i in range(hauteur):
        for j in range(largeur):
            genere_tuile(Carte, Liste_types[Tab_types[i,j]],taille_tuile*i , taille_tuile * j)

    plt.imshow(Carte.data, cmap='terrain')
    plt.show()
    """
    n,m = Carte.data.shape
    x = np.linspace(0,50,50)
    y = np.linspace(0,50,50)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    p = ax.plot_surface(X, Y, Carte.data[i * 50:i * 50 + 50,j * 50:j * 50 + 50].T, cmap ='winter')
    fig.colorbar(p)
    plt.show()"""

    show3d(Carte.data)
    #Carte.data=erosion(Carte.data)
    #output_path = "sim-final.png"
    #plt.imsave(output_path,Carte.data,cmap='gray')

