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
#On prend une carte avec les continents, on genere les fonds en 
#1-Donnant un type a chaque tuile
#2-Generant les fronti�res
#3-Generant les tuiles

taille_tuile = 50
hauteur = 10
largeur = 30

Carte = Map(-np.inf*np.ones((hauteur*taille_tuile,largeur*taille_tuile)))
Liste_types = [(-100,10,200)] #Listes de tous les types, et de ses caract�ristiques

def trouver_Ocean(Carte):
    #retourne les tuiles sur lesquelles il faut generer
    return Carte.indicator_grid(np.max,lambda x:np.float32(x.data <= 0),normaliser=False)

def attribution_types(Carte,Types):
    Tab_types = np.zeros((hauteur,largeur))
    return Tab_types

def genere_bords(Carte,Tab_types,Tuile_a_generer):
    pass

def genere_tuile(Carte,Tab_types,i,j,Tuile_a_generer):
    pass

if __name__=='__main__':
    Tuile_a_generer=trouver_Ocean(Carte)
    Tab_types=attribution_types(Carte,Liste_types)
    genere_bords(Carte,Tab_types,Tuile_a_generer)

    i,j=3,4
    genere_tuile(Carte,Tab_types,i,j,Tuile_a_generer)
    plt.imshow(Carte.data)
    plt.show()