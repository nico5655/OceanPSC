if __name__ == '__main__':
    import sys
    from pathlib import Path
    file = Path(__file__).resolve()
    parent, root = file.parent, file.parents[1]
    sys.path.append(str(root))

import time
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
    #retourne les tuiles sur lesquelles il faut generer
    return Carte.indicator_grid(np.max,lambda x:np.float32(x.data <= 0),normaliser=False)

def attribution_types(Carte,Types):
    Tab_types = np.zeros((hauteur,largeur))
    return Tab_types

def genere_bords(Carte,Tab_types,Tuile_a_generer):
    pass

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
     
        

if __name__ == '__main__':
    Tuile_a_generer = trouver_Ocean(Carte)
    Tab_types = attribution_types(Carte,Liste_types)
   
    genere_bords(Carte,Tab_types,Tuile_a_generer)
   
    i,j = 3,4
    genere_tuile(Carte,Tab_types,i,j,Tuile_a_generer,Liste_types)
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
    
