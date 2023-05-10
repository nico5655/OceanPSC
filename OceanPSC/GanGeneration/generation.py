from .CGAN import CGAN
from .params import *
import cv2
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter

def print_classes():
    for i,n in zip(range(len(class_names)),class_names):
        print(i,n)

class Generation:
    def __init__(self,inter_d=2):
        #instiating gan with training=False will auto load latest checkpoint and setup for max depth (i.e. 128x128)
        self.gan=CGAN(training=False,inter_d=inter_d)
        self.num_candidates=50
        self.inter_d=inter_d
        self.overlap_part=1/8

    def generate_one_sample(self,label,mean_stds=None):
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

        lat_size=2**(self.inter_d+2)
        overlap=max(int(lat_size*self.overlap_part),1)
        lat_size=lat_size-2*overlap
        
        big_map=np.zeros((size*(lat_size),size*(lat_size),128))
        for i in range(size):
            for j in range(size):
                if i==0 and j==0:
                    selected_img=self.gan([labels[i,j]],mstds=(means[i,j],stds[i,j]),output_intermediary=True)
                else:
                    potential_imgs=self.gan(self.num_candidates*[labels[i,j]],mstds=(means[i,j],stds[i,j]),output_intermediary=True).numpy()
                    if i==0:
                        sample_left=big_map[i*lat_size:i*lat_size+lat_size+overlap,j*lat_size:j*lat_size+overlap]
                        dist_left=np.sum((potential_imgs[:,overlap:,:overlap]-sample_left)**2,axis=(1,2,3))
                        dist_tot=dist_left
                    elif j==0:
                        sample_up=big_map[i*lat_size:i*lat_size+overlap,j*lat_size:j*lat_size+lat_size+overlap]
                        dist_up=np.sum((potential_imgs[:,:overlap,overlap:]-sample_up)**2,axis=(1,2,3))
                        dist_tot=dist_up
                    else:
                        if i==size-1:
                            sample_left=big_map[i*lat_size-overlap:i*lat_size+lat_size,j*lat_size:j*lat_size+overlap]
                            dist_left=np.sum((potential_imgs[:,:-overlap,:overlap]-sample_left)**2,axis=(1,2,3))
                        else:
                            sample_left=big_map[i*lat_size-overlap:i*lat_size+lat_size+overlap,j*lat_size:j*lat_size+overlap]
                            dist_left=np.sum((potential_imgs[:,:,:overlap]-sample_left)**2,axis=(1,2,3))
                        if j==size-1:
                            sample_up=big_map[i*lat_size:i*lat_size+overlap,j*lat_size-overlap:j*lat_size+lat_size]
                            dist_up=np.sum((potential_imgs[:,:overlap,:-overlap]-sample_up)**2,axis=(1,2,3))
                        else:
                            sample_up=big_map[i*lat_size:i*lat_size+overlap,j*lat_size-overlap:j*lat_size+lat_size+overlap]
                            dist_up=np.sum((potential_imgs[:,:overlap,:]-sample_up)**2,axis=(1,2,3))
                        dist_tot=dist_up+dist_left
                    arg=np.argmin(dist_tot)
                    selected_img=potential_imgs[arg]

                if j==size-1 and i==size-1:
                    big_map[i*lat_size:i*lat_size+lat_size,j*lat_size:j*lat_size+lat_size]=selected_img[overlap:-overlap,overlap:-overlap]
                elif j==size-1:
                    big_map[i*lat_size:i*lat_size+lat_size+overlap,j*lat_size:j*lat_size+lat_size]=selected_img[overlap:,overlap:-overlap]
                elif i==size-1:
                    big_map[i*lat_size:i*lat_size+lat_size,j*lat_size:j*lat_size+lat_size+overlap]=selected_img[overlap:-overlap,overlap:]
                else:
                    big_map[i*lat_size:i*lat_size+lat_size+overlap,j*lat_size:j*lat_size+lat_size+overlap]=selected_img[overlap:,overlap:]

        from .. import operations as op
        rslt=op.divide_treatment(big_map,lambda data: self.gan.call_on_intermediary(data).numpy(),r=4,out_multiplier=2**(5-self.inter_d),border=32)
        
        means2=cv2.resize(np.float32(means),(rslt.shape[0],rslt.shape[1]))
        stds2=cv2.resize(np.float32(stds),(rslt.shape[0],rslt.shape[1]))

        vrslt=rslt*stds2+means2
        return vrslt