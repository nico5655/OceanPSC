if __name__ == '__main__':
    import sys
    from pathlib import Path
    file = Path(__file__).resolve()
    parent, root = file.parent, file.parents[1]
    sys.path.append(str(root))

import numpy as np
from OceanPSC.Map import Map
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans

path="../data_plus_grande.tif"
reduce=(54,54)
nb_clusters=5

base_indicators=[(np.mean,None),(np.mean,lambda map: map.norme_grad),
                 (lambda x,axis: np.max(x,axis=axis)-np.min(x,axis=axis),None),
                 (np.mean,lambda x: x.data >=0)]

base_filt=np.array([1,1,1,1])

def map_clusters(mod_labels,labels,shape,reduce):
    rslt=np.zeros((shape[0]//reduce[0],shape[1]//reduce[1]))
    for k in range(len(mod_labels)):
        x,y=labels[k]
        rslt[x,y]=mod_labels[k]
    return rslt

def clusters(map,filt,nb_cluster):
    data,labels=map.get_indicators_data(base_indicators,reduce)
    data=data*filt
    model = KMeans(nb_cluster)
    model.fit(data, y=None, sample_weight= None)
    return map_clusters(model.labels_,labels,map.data.shape,reduce)


def main_clustering():
    map=Map.from_file(path)
    rsl=clusters(map,base_filt,nb_clusters)
    return rsl

if __name__ == '__main__':
    plt.imshow(main_clustering())
    plt.show()