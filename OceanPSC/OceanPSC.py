
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
 
<<<<<<< HEAD
# Generation d'un jeu de donnees al�atoire
#modele 0 : altitude moyenne, gradien moyen, amplitude d'altitude [moy,g,delta,x,y] puis les coord
#Coordonnees = [x,y] sur sur la terre, comment on g�re le baille sph�rique ?
#donn�es : entre altitude max et min [m,M], quelle proportion entre [m,am+bM] ect [..,M]
#donn�es : distanc p�le ou cmt g�r�er la d�formation de la projection ?

# Generation d'un jeu de donn�es al�atoire
#mod�le 0 : altitude moyenne, gradien moyen, amplitude d'altitude [moy,g,delta,x,y] puis les coord
#Coordonnees = [x,y] sur sur la terre, comment on gere le baille sph�rique ?
#donn�es : entre altitude max et min [m,M], quelle proportion entre [m,am+bM] ect [..,M]
#donn�es : distanc p�le ou cmt g�r�er la d�formation de la projection ?

# 
data = np.random.rand(10,5) #si on veut des coeffitients dans les donn�es il faut e faire avant -> rezize + coef car norme L2
nb_cluster = 3

#for nb_cluster in range(3,20):

model = MiniBatchKMeans(nb_cluster,init = 'k-means++',max_iter = 100,batch_size = 1024,compute_labels = True,n_init = 3,tol = 0.0)

#plus rapide 
#Pour d'autres algo https://scikit-learn.org/stable/modules/clustering.html#k-means

model.fit(data, y=None, sample_weight= None)

<<<<<<< HEAD
#model.labels_ tab donn�e i dans cluster model.labels_[j]
#cluster_centers_ tab des centroides et de leurs caract�ristiques
=======
model.labels_ #tab donn�e i dans cluster model.labels_[j]

>>>>>>> 5e8c93f428078b36d86ad972eff5280db8e75faf

#Les caract�ristiques de nos types sont les coordonn�es des centroides, il serait bien d'avoir des
#distributions satistiques pour choisir un point qui ne soit pas exactement le centroide -> plus tard

print(model.inertia_)






