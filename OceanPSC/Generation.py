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
from matplotlib.colors import LightSource

#On prend une carte avec les continents, on genere les fonds en
#1-Donnant un type a chaque tuile
#2-Generant les fronti�res
#3-Generant les tuiles
resolution = 7
taille_tuile = 2**resolution
hauteur = 5
largeur = 5
Carte = Map(-1000 * np.ones((largeur * taille_tuile,hauteur* taille_tuile)))
Liste_types = [(-1000,10,200),(-750,20,100),(-750,20,300),(-1100,20,100)] #Listes de tous les types, et de ses caract�ristiques alt moy, grad moy, amplitude min/max
def trouver_Ocean(Carte):
    # retourne les tuiles sur lesquelles il faut generer
    return Carte.indicator_grid(
        np.max, lambda x: np.float32(x.data <= 0), normaliser=False
    )


def attribution_types(Carte, Types):
    Tab_types = [
        [0] * hauteur for _ in range(largeur)
    ]  # il faut des int pas les float de numpy
    return Tab_types


def genere_bords(Carte, Tab_types, Tuile_a_generer, Liste_types):
    def altitude(l, h, T):
        alt = 0
        for t in T:
            alt += Liste_types[t][0] * (
                0.25 + (np.random.rand() * 2 - 1) / 20
            )  # choix de 0.05 arbitraire
        Carte.data[l * taille_tuile][h * taille_tuile] = alt

    def profil(l,h, t1, t2, dir):
        alt_moy = Liste_types[t1][0] * (
            0.5 + (np.random.rand() * 2 - 1) / 10
        ) + Liste_types[t2][0] * (0.5 + (np.random.rand() * 2 - 1) / 10)
        grad_moy = Liste_types[t1][1] * (
            0.5 + (np.random.rand() * 2 - 1) / 10
        ) + Liste_types[t2][1] * (0.5 + (np.random.rand() * 2 - 1) / 10)
        min_max = Liste_types[t1][2] * (
            0.5 + (np.random.rand() * 2 - 1) / 10
        ) + Liste_types[t2][2] * (0.5 + (np.random.rand() * 2 - 1) / 10)

    # Plus précisément, si la fonction est de classe C k \mathcal C^{k}, ses coefficients de Fourier sont négligeables devant 1 / n k {\displaystyle 1/n^{k}}
    # on genere un profil aleatoire par serie de fourier, avec un profil C2

    for h in range(hauteur):
        for l in range(largeur):
            t1 = Tab_types[l][h]
            t2 = Tab_types[l][h - 1]
            t3 = Tab_types[l - 1][h - 1]
            t4 = Tab_types[l - 1][h]
            altitude(l, h, (t1, t2, t3, t4))
            # on genere l'altitude en chacun des points intersecion de 4 tuiles a partir des types des 4 tuiles


    for h in range(hauteur):
        for l in range(largeur):
            t1 = Tab_types[l][h]
            t2 = Tab_types[l][h - 1]
            t4 = Tab_types[l - 1][h]
            profil(l, h, t1, t2, (1, 0))
            profil(l, h, t1, t4, (0, 1))



def genere_alt_moyennes(Carte,Tab_types):
    n,m=Tab_types.shape
    alt_moy=np.zeros((n,m))
    alt_moy_etendue=np.zeros((n+1,m+1))
    for k in range((len(Liste_types))):
        alt_moy[Tab_types==k]=Liste_types[k][0]
    alt_moy_etendue[:-1,:-1]=alt_moy
    alt_moy_etendue[-1,:]=alt_moy_etendue[-2,:]
    alt_moy_etendue[:,-1]=alt_moy_etendue[:,-2]

    def S(x):
        return 3 * x * x - 2 * x * x * x


    #Pseudo-random numbers, entre -1,1
    def a(i,j):
        return alt_moy_etendue[i,j]

    #Noise local : donne la valeur  en (x,y) de la fonction lisse qui relie les 4 points (i,j) ,...,(i+1,j+1)
    def N(x,y):

        i,j = np.int32(np.trunc(x)),np.int32(np.trunc(y))
        return a(i,j) + (a(i+1,j) - a(i,j)) * S(x - i) + (a(i,j+1) - a(i,j)) * S(y - j) + (a(i,j) - a(i,j+1) - a(i+1,j) + a(i+1,j+1)) * S(x - i) * S(y - j)

    I=np.arange(0,n*taille_tuile)
    J=np.arange(0,m*taille_tuile)
    I,J=np.meshgrid(I,J)
    Carte.data=N(I/taille_tuile,J/taille_tuile)
    plt.imshow(Carte.data,cmap='terrain')
    plt.show()


def genere_tuile(Carte,type_tuile,i_tuile,j_tuile):
    
    altitude_moyenne = type_tuile[0]
    gradient_moyen = type_tuile[1]
    amplitude_altitude = type_tuile[2]
        

    #Smooth Step function
    def S(x):
        return 3 * x * x - 2 * x * x * x

    Tab_reseau = 2*np.random.rand(taille_tuile*2,taille_tuile*2)-1


    #on pourrait mettre un facteur sqrt2 plutot que 2 dans le dimentionnement
    Terrain =  np.zeros((taille_tuile,taille_tuile))
    #Pseudo-random numbers, entre -1,1
    def a(i,j):
        return Tab_reseau[i,j]

    #Noise local : donne la valeur  en (x,y) de la fonction lisse qui relie les 4 points (i,j) ,...,(i+1,j+1)
    def N(x,y):

        i,j = np.int32(np.trunc(x)),np.int32(np.trunc(y))
        return a(i,j) + (a(i+1,j) - a(i,j)) * S(x - i) + (a(i,j+1) - a(i,j)) * S(y - j) + (a(i,j) - a(i,j+1) - a(i+1,j) + a(i+1,j+1)) * S(x - i) * S(y - j)

    #Génératrice du terrain en (i,j) de la tuile

    theta = np.pi/3
    cosinus = np.cos(theta)
    sinus = np.sin(theta)
    
    def f(i,j): #a valeur dans -2,2
        result = np.zeros((taille_tuile,taille_tuile))
        p = 1
        x = i/taille_tuile
        y = j/taille_tuile
        for  k in range(resolution):
            result += N(p * x,p * y) / p
            p *= 2
            x , y = 0.5 + cosinus * (x-0.5) - sinus * (y-0.5) , 0.5 + cosinus * (y-0.5) + sinus * (x-0.5)
            #rotation de centre le milieu du carré tuile
        return result
    
    def g(i,j):
        i_m,j_m = np.minimum(i,taille_tuile-i-1), np.minimum(j,taille_tuile-j-1)
        return i_m*j_m/(taille_tuile/2-1)**2

    I=np.arange(0,taille_tuile)
    J=np.arange(0,taille_tuile)
    I,J=np.meshgrid(I,J)

    i_carte = i_tuile + I
    j_carte = j_tuile + J
    Carte.data[i_carte,j_carte] +=  0.25*amplitude_altitude * f(I,J)*g(I,J) #+bonne moyenne continue

#le résultat n'est pas incroyable, peut etre il faut garder les ^mêmes nombre aléatoire et simplement en rajouter ?

def show3d(map2):
    x = np.linspace(0, hauteur * taille_tuile, hauteur * taille_tuile)
    y = np.linspace(0, largeur * taille_tuile, largeur * taille_tuile)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(map2, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(X,Y, map2, rstride=5, cstride=5, facecolors=rgb,
                           linewidth=0, antialiased=False, shade=True)
    fig.colorbar(surf)
    plt.show()


if __name__ == "__main__":
    Tuile_a_generer = trouver_Ocean(Carte)
    #Tab_types = attribution_types(Carte, Liste_types)

    genere_tuile(Carte, Liste_types[0], 0, 0)
    Tab_types=np.int32(np.trunc(np.random.rand(5,5)*len(Liste_types)))
    genere_alt_moyennes(Carte,Tab_types)

    for i in range(9):
        for j in range(9):
            genere_tuile(Carte, Liste_types[Tab_types[i//2,j//2]],taille_tuile//2*i , taille_tuile//2*j)

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

