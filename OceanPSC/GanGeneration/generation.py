from .CGAN import CGAN
from .params import *
import cv2

def print_classes():
    for i,n in zip(range(len(class_names)),class_names):
        print(i,n)

class Generation:
    def __init__(self):
        #instiating gan with training=False will auto load latest checkpoint and setup for max depth (i.e. 128x128)
        self.gan=CGAN(training=False)

    def generate_one_sample(self,label,mean_std=None):
        """Generate one 128x128 elevation map (in meters) sample of given label (str or int).
        Meanstd should tuple, list or nparray (the meters value shouldn't be too far away from expected value range for the label),
        leaving None will result in using default mean_std for this label."""
        if type(label) is str:
            label=class_names.index(label)
        return self.gan(label,mstds=mean_stds)

    def generate_tiled_map(self,labels,means=None,stds=None):
        """Generate a (sizex128,sizex128) elevation map in meters. labels, means and stds should be (size,size) grids.
        Means and stds shouldn't be too far away from usual range for matching label.
        Defaults for means and stds are calculated using label. Avoid doing this for means unless you only have one label.
        Avoid using a means map with an important variance, or a label map that would result in such a map."""

        size=labels.shape[0]
        if means is None:
            means=np.zeros((size,size))
            for k in range(size):
                for l in range(size):
                    means[k,l]=vals[labels[k,l]][0]*5000
                    
        if stds is None:
            stds=np.zeros((size,size))
            for k in range(size):
                for l in range(size):
                    stds[k,l]=vals[labels[k,l]][1]*1000


        means2=cv2.resize(np.float32(means),(size*128,size*128))
        stds2=cv2.resize(np.float32(stds),(size*128,size*128))

        big_map=np.zeros((size*4,size*4,130))
        for i in range(size):
            for j in range(size):
                big_map[i*4:i*4+4,j*4:j*4+4,:]=self.gan(labels[i,j],mstds=(means[i,j],stds[i,j]),output_intermediary=True)

        rslt=self.gan.call_on_intermediary(big_map).numpy()
        vrslt=rslt*stds2+means2
        return vrslt