
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
 
# Génération d'un jeu de donnees aléatoire
#modèle 0 : altitude moyenne, gradien moyen, amplitude d'altitude [moy,g,delta,x,y] puis les coord
#Coordonnees = [x,y] sur sur la terre, comment on gère le baille sphérique ?
#données : entre altitude max et min [m,M], quelle proportion entre [m,am+bM] ect [..,M]
#données : distanc pôle ou cmt géréer la déformation de la projection ?
# 
data = np.random.rand(10,5) #si on veut des coeffitients dans les données il faut e faire avant -> rezize + coef car norme L2
nb_cluster = 3

#for nb_cluster in range(3,20):

model = MiniBatchKMeans(nb_cluster,init = 'k-means++',max_iter = 100,batch_size = 1024,compute_labels = True,n_init = 3,tol = 0.0)

#plus rapide 
#Pour d'autres algo https://scikit-learn.org/stable/modules/clustering.html#k-means

model.fit(data, y=None, sample_weight= None)

#model.labels_ tab donnée i dans cluster model.labels_[j]
#cluster_centers_ tab des centroides et de leurs caractéristiques

#Les caractéristiques de nos types sont les coordonnées des centroides, il serait bien d'avoir des
#distributions satistiques pour choisir un point qui ne soit pas exactement le centroide -> plus tard

print(model.inertia_)






