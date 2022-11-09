#Algorithme MCMC pour sélectionner un état

T = 200 #une température initiale aléatoire
s = initialState()
t=Transitions(s) # Les n transitions possibles 
seuil = 0.01 #Critère de convergence (en température)

#Itérer jusqu'au critère de convergence
while T > seuil:

  #Calculer l'état courant du coût 
    curcost = costfunc(s)
    newcost= [0]*n     #Initialiser newcost à 0
    probability=[0]*n #Initialiser les probabilités de transitions à 0

    #Calculuer les Probabilités
    for i in range(n):
        newcost[i] = costfunc(doTransition(s, t[i]))
        probability[i] = exp(-(newcost[i] - curcost)/T)
        
  #Normaliser les Probabilités
  probability /= sum(probability)

  #Sélectionner la Transition
  sampled = sample_transition(t, probability)

  #Faire la transition
  s = doTransition(s, sampled)

  #Baisser la Température
  T *= 0.975
