import numpy as np

#On prend une carte avec les continents, on genere les fonds en 
#1-Donnant un type a chaque tuile
#2-Generant les frontières
#3-Generant les tuiles

taille_tuile = 50
hauteur = 10
largeur = 30

Carte = Map(-1*np.ones((hauteur*taille_tuile,largeur*taille_tuile)))
Liste_types = [(-100,10,200)] #Listes de tous les types, et de ses caractéristiques

def trouver_Ocean(Carte):
    #retourne les tuiles sur lesquelles il faut generer
    return Carte.indicator_grid(np.max,lambda x: x.data <=0)

def attribution_types(Carte,Types):
    Tab_types = np.zeros((hauteur,largeur))
    return Tab_types

def genere_bords(Carte,Tab_types,Tuile_a_generer):


def genere_tuile(Carte,Tab_types,i,j):





