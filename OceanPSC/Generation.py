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

#On prend une carte avec les continents, on genere les fonds en
#1-Donnant un type a chaque tuile
#2-Generant les fronti�res
#3-Generant les tuiles
taille_tuile = 50
hauteur = 10
largeur = 30

Carte = Map(-np.inf * np.ones((hauteur * taille_tuile,largeur * taille_tuile)))
Liste_types = [(-100,10,200)] #Listes de tous les types, et de ses caract�ristiques
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
            )  # choix de 0.1 arbitraire
        Carte.data[l * taille_tuile][h * taille_tuile] = alt

    def profil(h, l, t1, t2, dir):
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

def genere_tuile(Carte,Tab_types,i,j,Tuile_a_generer,Liste_types):
    
    indice = int(Tab_types[i][j])
    altitude_moyenne = Liste_types[indice][0]
    gradient_moyen = Liste_types[indice][1]
    amplitude_altitude = Liste_types[indice][2]
        

    #Smooth Step function
    def S(x):
        return 3 * x * x - 2 * x * x * x

    #Pseudo-random numbers
    def a(i,j):
        u,v = 50 * (i / np.pi - np.floor(i / np.pi)),50 * (j / np.pi - np.floor(j / np.pi))
        return 2 * (u * v * (u + v) - np.floor(u * v * (u + v))) - 1

    def am(i:int,j:int):
        ###il faut ajouter une ligne pour agir sur les coefficients sur les
        ###bords et sur des points à l'intérieur de la tuile
        if i % taille_tuile == 0 or j % taille_tuile == 0 or Carte.data[int(i // taille_tuile)][int(j // taille_tuile)] >= 0:
            return max(1,(Carte.data[i / taille_tuile,j / taille_tuile] - altitude_moyenne) / amplitude_altitude)
        ###n'oublie pas les bords périodiques
        u,v = 50 * (i / np.pi - np.floor(i / np.pi)),50 * (j / np.pi - np.floor(j / np.pi))
        return 2 * (u * v * (u + v) - np.floor(u * v * (u + v))) - 1


    #Conditions de raccordement
    def b(i,j): return a(i + 1,j)
    def c(i,j): return a(i,j + 1)
    def d(i,j): return a(i + 1,j + 1)

    def bm(i,j): return am(i + 1,j)
    def cm(i,j): return am(i,j + 1)
    def dm(i,j): return am(i + 1,j + 1)

    #Noise local
    def N(x,y):
        i,j = np.floor(x),np.floor(y)
        return a(i,j) + (b(i,j) - a(i,j)) * S(x - i) + (c(i,j) - a(i,j)) * S(y - j) + (a(i,j) - b(i,j) - c(i,j) + d(i,j)) * S(x - i) * S(y - j)

    #Noise local modifié
    def Nm(x,y):
        i,j = np.floor(x),np.floor(y)
        return am(i,j) + (bm(i,j) - am(i,j)) * S(x - i) + (cm(i,j) - am(i,j)) * S(y - j) + (am(i,j) - bm(i,j) - cm(i,j) + dm(i,j)) * S(x - i) * S(y - j)




    temp = np.zeros((hauteur * taille_tuile,largeur * taille_tuile))
    for dx in range(taille_tuile):
        for dy in range(taille_tuile):
            x = taille_tuile * i + dx
            y = taille_tuile * j + dy
            temp[x][y] = .5 * (amplitude_altitude * Nm(x / taille_tuile,y / taille_tuile) + altitude_moyenne)

    #Génératrice du terrain
    def f(x,y,iterations):
        result = 0
        p = 1
        for i in range(1,iterations):
            result+=N(p * x,p * y) / p
            p*=2
            temp = x
            x = 3 / 5 * x - 4 / 5 * y
            y = 4 / 5 * temp + 3 / 5 * y
        return result

    for dx in range(taille_tuile):
        for dy in range(taille_tuile):
            x = taille_tuile * i + dx
            y = taille_tuile * j + dy
            if Carte.data[x,y] == -np.inf:
                Carte.data[x,y] = temp[x][y] + .5 * (amplitude_altitude * f(x / taille_tuile,y / taille_tuile,1) + altitude_moyenne)


def show3d(map2):
    x = np.linspace(0, hauteur * taille_tuile, hauteur * taille_tuile)
    y = np.linspace(0, largeur * taille_tuile, largeur * taille_tuile)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlim3d(0, hauteur * taille_tuile)
    ax.set_ylim3d(0, largeur * taille_tuile)
    p = ax.plot_surface(X, Y, map2, cmap="winter")
    fig.colorbar(p)
    plt.show()


if __name__ == "__main__":
    Tuile_a_generer = trouver_Ocean(Carte)
    Tab_types = attribution_types(Carte, Liste_types)
    genere_bords(Carte, Tab_types, Tuile_a_generer, Liste_types)
    i, j = 3, 4
    genere_tuile(Carte, Tab_types, i, j, Tuile_a_generer, Liste_types)
    plt.imshow(Carte.data)
    plt.show()

    n,m = Carte.data.shape
    x = np.linspace(0,50,50)
    y = np.linspace(0,50,50)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    p = ax.plot_surface(X, Y, Carte.data[i * 50:i * 50 + 50,j * 50:j * 50 + 50].T, cmap ='winter')
    fig.colorbar(p)
    plt.show()
    
    plt.show()
    show3d(Carte.data)
