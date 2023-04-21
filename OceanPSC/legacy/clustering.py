if __name__ == '__main__':
    import sys
    from pathlib import Path
    file = Path(__file__).resolve()
    parent, root = file.parent, file.parents[1]
    sys.path.append(str(root))

import numpy as np
import matplotlib.pyplot as plt
import OceanPSC.operations as op
from OceanPSC.Map import Map
from skimage.measure import block_reduce
import copy
import OceanPSC.DEM as d
import OceanPSC.classification as c
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
import pandas as pd

def transfo_majority(dta,axis):
    new=np.zeros((dta.shape[0],dta.shape[1]))
    for i in range(dta.shape[0]):
        for j in range(dta.shape[1]):
            selec=dta[i,j]
            new[i,j]=np.bincount(selec.reshape(-1)).argmax()
    return new

def transfo_density(data,axis):
    print(axis)
    result=np.zeros((data.shape[0],data.shape[1],8))
    for k in range(8):
        result[:,:,k]=(data==k).mean(axis=axis)
    return result

def get_adjacency(n,m):
    size=n*m
    adjacency=np.zeros((size,size),dtype=np.bool)
    for i in range(n):
        for j in range(m):
            for k in range(n):
                for l in range(m):
                    ind=i*m+j
                    ind2=k*m+l
                    if abs(i-k)<=1 and abs(j-l)<=1:
                        adjacency[ind,ind2]=adjacency[ind2,ind]=True
    return adjacency

def reduction_clustering(classes,elev):
    fc=classes.copy()
    fc[fc==6]=5
    fc[fc>8]=7
    fc[fc>6]=fc[fc>6]-1
    relev=block_reduce(elev,(50,50),np.mean)
    intol=block_reduce(elev,(50,50),np.min)
    fclr=block_reduce(np.int32(fc),(50,50),transfo_density)
    fc2=classes.copy()
    rs=block_reduce(np.int32(fc2),(50,50),transfo_majority)
    adjacency=get_adjacency(fclr.shape[0],fclr.shape[1])
    A,B=np.meshgrid((intol<=0).flatten(),(intol<=0).flatten())
    adjacency=adjacency[A&B].reshape((intol<=0).sum(),(intol<=0).sum())
    vals=AgglomerativeClustering(n_clusters=1000,connectivity=adjacency,linkage='ward').fit(fclr[intol<=0]).labels_
    vals11=-np.ones(fclr.shape[:-1])
    vals11[intol<=0]=vals
    vals=vals11
    clusts=np.zeros((1000,8))
    for k in range(1000):
        clusts[k]=fclr[vals==k,:].mean(axis=0)
    distance_matrix = pdist(clusts)

    clusters = linkage(distance_matrix, 'ward')
    dendrogram(clusters, p=8,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    truncate_mode='level',
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  orientation='top')

    rslt=fcluster(clusters,3.35,criterion='distance')
    rslt=rslt-1
    nc=int(np.max(rslt))
    vals3=vals.copy()
    for k in range(512):
        vals3[vals==k]=rslt[k]

    vals3=np.float32(vals3)    
    vals3[intol>0]=np.nan
    aa=[]
    aa.append(['MOR','VRS','RS','AP','CR','CS','CSH','SCA'])
    for k in range(nc+1):
        aa.append(np.int32(fclr[(vals3==k)].mean(axis=0)*100))
    df=pd.DataFrame(columns=aa[0],data=aa[1:])
    return df, vals3,clusters,vals

def main_cls():
    return reduction_clustering(np.load('data/classes.npy'),np.load('data/st.npy')[250:-250,:])