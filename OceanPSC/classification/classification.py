import matplotlib.colors as c
import numpy as np
from .. import operations as op
from ..Map import Map
import cv2
from .DEM import DEM
from skimage.measure import block_reduce

#everything below tresh is of class
#slope
flat_tresh=0.9 # 1:64
sloping_tresh=1.9 # 1:32
steep_tresh=3.6# 1:16
#roughness
smooth_thresh=0.25
rough_tresh=0.50
#organization
disorganized_treshold=0.73
organized_treshold=0.82
#elevation
abyssal_tresh=-4600
deep_tresh=-3600
intermediate_tresh=-1600

#identifiers
MID_OCEANIC_RIDGE=0
VERY_ROUGH_SEAFLOOR=1
ROUGH_SEAFLOOR=2
ABYSSAL_PLAIN=3
CONTINENTAL_RISE=4
CONTINENTAL_SLOPE=5
CONTINENTAL_SHELF=6
SCARPS=7
LAND=8
SEAMOUNT=9

def clean(cls_map,dist):
    """Perform cleaning via a (2dist+1)x(2dist+1) majority filter"""
    rslt=cls_map.copy()
    vois=op.neighbor_grid(rslt,dist,bound_method="duplicate")
    max_decider=np.zeros((rslt.shape))
    arg_max_decider=rslt.copy()
    for k in range(int(np.min(cls_map)),int(np.max(cls_map))+1):
        decider=(vois==k).sum(axis=(-1,-2))
        arg_max_decider[decider>max_decider]=k
        arg_max_decider[decider==max_decider]=rslt[decider==max_decider]
        max_decider[decider>max_decider]=decider[decider>max_decider]
    return arg_max_decider
    
def get_sea_classification(slope,roughness,organization,elevation):
    """Carte de classification sous-marine du terrain basée sur les indicateurs."""

    ###STEP 1: creation of the base classification (using local indicators for each point)
    ## STEP 1.1: terrain classification
    flat=slope<flat_tresh
    sloping=(slope<sloping_tresh)&(~flat)
    steep=(slope<steep_tresh)&(~(sloping|flat))
    scarped=~(flat|sloping|steep)

    smooth=(roughness<smooth_thresh)
    rough=(roughness<rough_tresh)&(~smooth)
    very_rough=~(smooth|rough)

    disorganized=organization<disorganized_treshold
    organized=(organization<organized_treshold)&(~disorganized)
    very_organized=~(disorganized|organized)

    ## STEP 1.2: terrain regions
    SMOOTH=flat&smooth&(disorganized|organized)
    ORGANIZED=flat&((smooth&very_organized)|(rough&(organized|very_organized)))
    SLOPING=(sloping&(smooth|very_organized))|(steep&(smooth|(rough&(organized|very_organized))))
    
    ROUGH=((flat|sloping|steep)&rough&disorganized)|(sloping&rough&organized)
    VERY_ROUGH=(flat|(sloping&disorganized))&very_rough
    SCARPED=~(SMOOTH|ORGANIZED|SLOPING|ROUGH|VERY_ROUGH)
    
    regions=np.zeros(elevation.shape)
    regions[SMOOTH]=0
    regions[ORGANIZED]=1
    regions[SLOPING]=2
    regions[SCARPED]=3
    regions[ROUGH]=4
    regions[VERY_ROUGH]=5

    #sea terrain consolidation
    cos=np.float32(regions.copy())
    #Land is treated as SMOOTH terrain for consolidation in this step
    #the region classification is only accurate for sea and its labels can't be used for land
    #the consolidation only uses land that borders ocean regions in order to consolidate those regions
    #and given that the sea adjacent to land is usually smooth (see continental_shelf class), it makes sense to classify coastal land as smooth
    cos[elevation>=0]=0
    cos=clean(cos,1)
    cos=clean(cos,2)
    cos=clean(cos,3)
    cos=clean(cos,4)
    regions[elevation<0]=cos[elevation<0]

    SMOOTH=(regions==0)
    ORGANIZED=(regions==1)
    SLOPING=(regions==2)
    SCARPED=(regions==3)
    ROUGH=(regions==4)
    VERY_ROUGH=(regions==5)

    ## STEP 1.3: classes
    #auxiliary variables
    abyssal=elevation<abyssal_tresh
    deep=(elevation<deep_tresh)&(~abyssal)
    intermediate=(elevation<intermediate_tresh)&(~(abyssal|deep))
    shallow=~(abyssal|intermediate|deep)

    #classification
    Abyssal_Plain=SMOOTH&(intermediate|deep|abyssal)
    Continental_Shelf=(SMOOTH|ORGANIZED|ROUGH)&shallow
    Continental_Rise=ORGANIZED&(intermediate|deep|abyssal)
    Slopes=SLOPING
    Scarps=SCARPED
    Rough_Seafloor=ROUGH&(deep|abyssal)
    Mid_Oceanic_Ridge=(ROUGH&intermediate)|(VERY_ROUGH&(shallow|intermediate))
    Very_Rough_Seafloor=VERY_ROUGH&(deep|abyssal)

    #result
    classes=np.zeros(elevation.shape)
    classes[Abyssal_Plain]=ABYSSAL_PLAIN
    classes[Continental_Shelf]=CONTINENTAL_SHELF
    classes[Continental_Rise]=CONTINENTAL_RISE
    classes[Slopes]=CONTINENTAL_SLOPE
    classes[Scarps]=SCARPS
    classes[Rough_Seafloor]=ROUGH_SEAFLOOR
    classes[Mid_Oceanic_Ridge]=MID_OCEANIC_RIDGE
    classes[Very_Rough_Seafloor]=VERY_ROUGH_SEAFLOOR

    #consolidation of classes by majority filter
    #land is treated as its close substitute continental_shelf
    #land labels are only used for classification of land adjacent element, i.e continental_shelf
    classes[elevation>=0]=CONTINENTAL_SHELF
    classes=clean(classes,1)
    classes=clean(classes,4)

    classes=np.float32(classes)
    classes[(elevation>=0)&(classes==CONTINENTAL_SHELF)]=LAND

    ###STEP 2: classification enhancement (based on overall consistency measurement and class interpretation)

    def propag_on(start,target,dist=0):
        """Expansion of binary image 'start' to all points connected to 'start' in 'target'. Eventuallly with a maximum distance."""
        cpt=0
        old=-1
        validation=start.copy()
        if target is None:
            target=np.ones_like(classes,dtype=np.bool)
        while True:
            filt=np.ones((3,3))
            filt[0,0]=filt[2,2]=filt[0,2]=filt[2,0]=0
            neighborhood=(op.neighbor_grid(validation)*filt).sum(axis=(-1,-2))>0
            validation[(target)&(neighborhood)]=1
            #round earth fix
            validation[:,0][((validation[:,-1]==1)&(target[:,0]))]=1
            if cpt%10==0:
                if old==validation.sum():
                    break
                old=validation.sum()
            cpt+=1
            if (dist>0) and (cpt>dist):
                break
        return validation

    def consolidate_numbers(marqued):
        """Label each connected component of a binary image. (each connected groupe of ones has a label, and all zeros are labeled zero)"""
        def step_numbers(direction,dmar):
            """direction: horizontal +:1, -:2, vertical +: 3, vertical -: 4"""
            cpt=0
            if direction == 1 or direction == 2:
                eq_next=(dmar[:,1:]!=0)&(dmar[:,:-1]!=0)
            else:
                eq_next=(dmar[1:,:]!=0)&(dmar[:-1,:]!=0)
            while True:
                pre_numbers=dmar.copy()
                if direction ==1:
                    dmar[:,1:][eq_next]=dmar[:,:-1][eq_next]
                elif direction == 2:
                    dmar[:,:-1][eq_next]=dmar[:,1:][eq_next]
                elif direction == 3:
                    dmar[1:,:][eq_next]=dmar[:-1,:][eq_next]
                else:
                    dmar[:-1,:][eq_next]=dmar[1:,:][eq_next]
                if cpt % 100 == 0:
                    m=(pre_numbers!=dmar).mean()
                    if m==0:
                        break
                cpt+=1

        dar=np.zeros_like(classes)
        dar[marqued==1]=1+np.arange(int(marqued.sum()))
        for k in range(10):
            old=len(np.unique(dar))
            randx=np.random.rand()>0.5
            randy=np.random.rand()>0.5
            s1=1+int(randx)
            s2=3+int(randy)
            step_numbers(s1,dar)
            step_numbers(s2,dar)
            nov=len(np.unique(dar))
            if old==nov:
                break

        eq_bord=(dar[:,-1]!=0)&(dar[:,0]!=0)
        dar[:,0][eq_bord]=dar[:,-1][eq_bord]

        while True:
            cpt=0
            filtr=np.ones((3,3))
            filtr[0,0]=filtr[2,2]=filtr[2,0]=filtr[0,2]=0
            filtr[1,1]=0
            dt=np.int32(op.neighbor_grid(dar)*filtr)
            ls=np.unique(dar)
            prev=len(ls)
            for k in ls:
                if k!=0 and (dar==k).sum()>0:
                    dta=dt[dar==k]
                    js=np.unique(dta[(dta!=k)&(dta!=0)])
                    for j in js:
                        if j>k:
                            dar[dar==j]=k
            fin=len(np.unique(dar))
            if fin==prev:
                break
        return dar

    def wipe(clas,dar,deletion=None):
        """Remove classes of all labeled components and replace each component by the biggest adjacent class.
        'dar' should contain non-zeros for all areas that need to be removed and the same number for all points in the same connected component."""
        if deletion is None:
            deletion=dar
        for k in np.unique(deletion):
            if k!=0:
                opc=op.neighbor_grid(classes)[dar==k].flatten()
                opc=opc[~big_categories[clas](opc)]
                arg=np.argmax(np.bincount(np.int32(opc)))
                classes[(dar==k)&big_categories[clas](classes)]=arg

    
    ## STEP 2.1: adjacency filtering (reclassify areas using the interpretation of classes)
    #i.e: CONTINENTAL_*** should be connected to land and MID_OCEANIC_RIDGE should be in the ocean
    def do_cont_shelf():
        """Remove all continental_shelf that isn't adjacent to land."""
        propagation=propag_on(big_categories['LAND'](classes),big_categories['CONTINENTAL_SHELF'](classes))
        marqued=(1-propagation)*(big_categories['CONTINENTAL_SHELF'](classes))
        wipe('CONTINENTAL_SHELF',consolidate_numbers(marqued))
        print('csh cleaned!')

    #all mid_oceanic ridge should be, well ... in the ocean (i.e. adjacent to large rough seafloor regions)
    img=np.float32(big_categories['ROUGH_SEAFLOOR'](classes))
    kernel = np.ones((6,6),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=5)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=15)
    propagation=propag_on(img==1,big_categories['MID_OCEANIC_RIDGE'](classes))

    def remove_bad_mor(rem):
        """replacing removed mid_oceanic ridge by most likely alternative"""
        classes[big_categories['MID_OCEANIC_RIDGE'](classes)&(rem)&(~flat)]=CONTINENTAL_SLOPE
        classes[big_categories['MID_OCEANIC_RIDGE'](classes)&(rem)&(flat)&(shallow)]=CONTINENTAL_SHELF
        classes[big_categories['MID_OCEANIC_RIDGE'](classes)&(rem)&(flat)&(~shallow)&(disorganized)]=ABYSSAL_PLAIN
        classes[big_categories['MID_OCEANIC_RIDGE'](classes)&(rem)&(flat)&(~shallow)&(~disorganized)]=CONTINENTAL_RISE

    #bad mor are replaced based on most likely alternative
    remove_bad_mor(propagation==0)
    #bad csh are replaced by best neighboor
    do_cont_shelf()
    #csh doesn't have a good 'most likely alternative'

    def do_cont_slope():
        """Reclassify continental_slope that isn't close to the continent"""
        #criteria 1: connectivity to the continent
        propagation=propag_on(big_categories['LAND'](classes)|big_categories['CONTINENTAL_SHELF'](classes),
                            big_categories['CONTINENTAL_SLOPE'](classes))
        marqued=(1-propagation)*(big_categories['CONTINENTAL_SLOPE'](classes))
        classes[marqued==1]=SEAMOUNT
        #criteria 2: closeness to the continent
        propagation=propag_on(big_categories['LAND'](classes)|big_categories['CONTINENTAL_SHELF'](classes),
                                None,100)
        marqued=(1-propagation)*(big_categories['CONTINENTAL_SLOPE'](classes))
        #a new class is created for non-continental slopes
        classes[marqued==1]=SEAMOUNT
        print('cslope cleaned!')

    def do_cont_rise():
        """Reclassify continental_rise that isn't close to the continent"""
        propagation=propag_on(
            big_categories['LAND'](classes)|big_categories['CONTINENTAL_SHELF'](classes)|big_categories['CONTINENTAL_SLOPE'](classes),
            big_categories['CONTINENTAL_RISE'](classes)
        )
        #removal based on most likely alternative
        classes[big_categories['CONTINENTAL_RISE'](classes)&(propagation==0)&smooth]=ABYSSAL_PLAIN
        classes[big_categories['CONTINENTAL_RISE'](classes)&(propagation==0)&(~smooth)&(~intermediate)]=ABYSSAL_PLAIN
        classes[big_categories['CONTINENTAL_RISE'](classes)&(propagation==0)&(very_rough|rough)&(intermediate)]=MID_OCEANIC_RIDGE
        print('c_rise cleaned!')

     #reclassifying continental_slope that isn't "continental" as "seamount"
    do_cont_slope()
    #reclassifying continental_rise that isn't "continental" using most likely alternative
    do_cont_rise()

    ## STEP 2.2: MID_OCEANIC_RIDGE treatment and removing irrelevant local features (method is based on class interpretation)
    #local intermontain basis in MOR are classified as abyssal_plain, but they are globally in MOR
    img=np.float32(big_categories['MID_OCEANIC_RIDGE'](classes))
    kernel = np.ones((5,5),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=5)
    classes[(img==1)&(classes==ABYSSAL_PLAIN)]=MID_OCEANIC_RIDGE
    
    #MOR is an elevation in the middle of the rough seafloor, its main adjacency should be rough seafloor
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel, iterations=15)
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, iterations=25)
    classes[(img==0)&(classes==MID_OCEANIC_RIDGE)]=42
    dmar=consolidate_numbers(classes==42)
    for k in np.unique(dmar):
        if k!=0:
            opc=op.neighbor_grid(classes)[dmar==k].flatten()
            opc=opc[opc!=42]
            arg=np.argmax(np.bincount(np.int32(opc)))
            if big_categories['ROUGH_SEAFLOOR'](arg):
                arg=0
            classes[dmar==k]=arg
    
    oldc=classes.copy()
    #majority filter consolidation
    classes=clean(classes,1)
    classes=clean(classes,2)
    classes[oldc==LAND]=LAND

    #parts of mid_oceanic_ridge are sloping (and therefore classified as seamount given their non adjacency to land)
    #the non-seamount sloping areas are linked to land areas (islands on the ridge or continents close to the ridge)
    img=np.float32(big_categories['MID_OCEANIC_RIDGE'](classes))
    kernel = np.ones((6,6),np.uint8)
    #consolidation is done around large and confirmed MID_OCEANIC_RIDGE regions
    #indeed, the missclassification, goes both way and scattered MOR are present in SEAMOUNT areas
    #the operations below create a binary image representing the 'main' ridges and their surroundings
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=5)
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel, iterations=3)
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7)), iterations=10)
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7)), iterations=30)
    img=np.pad(img,1000,constant_values=0)#edge management
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=90)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7)), iterations=60)
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, iterations=10)
    img=img[1000:-1000,1000:-1000]
    #this consolidation avoids reclassifying consolidated real seamounts that happens to be close to the ridge
    img=np.float32((img==1)&((classes==MID_OCEANIC_RIDGE)|(classes==SEAMOUNT)))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=6)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=5)
    classes[(img==1)&(classes==SEAMOUNT)]=MID_OCEANIC_RIDGE

    #full ridge consolidation in order to increase connectivity
    img=np.float32(big_categories['MID_OCEANIC_RIDGE'](classes))
    kernel = np.ones((6,6),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=6)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7)), iterations=7)

    #MOR are very consolidated and connected, so all scattered components are removed
    #the previous map of large component and surroundings, with the addition of reclassified seamounts is used to filter all MOR
    marqued=(img==1)|(big_categories['MID_OCEANIC_RIDGE'](classes))
    dmar=consolidate_numbers(marqued)
    def reduce_nums(dmar):
        sizes=[]
        ls=np.unique(dmar)
        for k in range(len(ls)):
            sizes.append((dmar==ls[k]).sum())
            dmar[dmar==ls[k]]=k
        sizes=np.array(sizes)
        return sizes


    sizes=reduce_nums(dmar)
    dmar=np.int32(dmar)
    deletion=dmar[sizes[dmar]<=100]

    #small components are removed via the adjacency method
    wipe('MID_OCEANIC_RIDGE',dmar,deletion)

    for k in np.unique(deletion):
        dmar[dmar==k]=0

    #larger components (but not large enough) are removed via "most likely alternative" method
    sizes=reduce_nums(dmar)
    dmar=np.int32(dmar)
    rem=(sizes[np.int32(dmar)]<=70000)

    remove_bad_mor(rem)

    img=np.float32(big_categories['MID_OCEANIC_RIDGE'](classes))
    kernel = np.ones((5,5),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
    classes[img==1]=MID_OCEANIC_RIDGE
    
    #removing the small outliers
    marqued=(big_categories['MID_OCEANIC_RIDGE'](classes))
    dmar=consolidate_numbers(marqued)
    sizes=reduce_nums(dmar)
    dmar=np.int32(dmar)
    deletion=dmar[sizes[dmar]<=250]
    wipe('MID_OCEANIC_RIDGE',dmar,deletion)

    #the adjacency filtering is reused on the reclassification
    do_cont_shelf()
    do_cont_slope()
    do_cont_rise()

    #majority filter consolidation
    classes=clean(classes,1)
    classes=clean(classes,2)

    pre_large_scale=classes.copy()

    ## STEP 2.3: large scale consolidation on seafloor (abyssal_plain, rough_seafloor, very_rough_seafloor)
    #ABYSSAL_PLAIN are consolidated into large areas
    img=np.float32(big_categories['ABYSSAL_PLAIN'](classes))
    kernel = np.ones((5,5),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=5)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=5)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=15)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=15)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=50)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=25)
    classes[(img==1)&(big_categories['ROUGH_SEAFLOOR'](classes))]=ABYSSAL_PLAIN

    #non-consolidated plains are removed (and replaced by rough_seafloor, its most likely alternative)
    img=np.float32(big_categories['ROUGH_SEAFLOOR'](classes)|big_categories['MID_OCEANIC_RIDGE'](classes))
    kernel = np.ones((5,5),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=3)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=11)
    classes[(img==1)&(big_categories['ABYSSAL_PLAIN'](classes))]=ROUGH_SEAFLOOR

    #removing small seamounts in the middle of continental_rise (except for large structures, there shouldn't be "sea"mounts in these areas)
    img=np.float32(classes==CONTINENTAL_RISE)
    kernel = np.ones((5,5),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=5)
    classes[(img==1)&(classes==SEAMOUNT)]=CONTINENTAL_RISE
    

    ## STEP 2.4: final consolidation by reducing detail level on the class map
    def dens(cla,axis):
        rsl=np.zeros((cla.shape[0],cla.shape[1],11))
        for i in range(11):
            rsl[:,:,i]=(cla==i).mean(axis=axis)
        return rsl


    cll=block_reduce(classes,(5,5),dens)
    fin_c=np.argmax(cll,axis=-1)
    fin_c[(cll[:,:,VERY_ROUGH_SEAFLOOR]-cll[:,:,ABYSSAL_PLAIN]-cll[:,:,CONTINENTAL_RISE])>0.35]=VERY_ROUGH_SEAFLOOR

    cl2c=fin_c.copy()
    fin_c=clean(fin_c,1)
    fin_c[cl2c==LAND]=LAND

    final_classes=np.zeros_like(classes)
    I,J=np.indices(final_classes.shape)
    I,J=block_reduce(I,(5,5),lambda x,axis: x),block_reduce(J,(5,5),lambda x,axis: x)
    for k in range(SEAMOUNT+1):
        final_classes[I[fin_c==k,:,:],J[fin_c==k,:,:]]=k

    return final_classes



big_categories={
    'LAND': lambda cla: cla==LAND,
    'CONTINENTAL_SHELF': lambda cla: cla==CONTINENTAL_SHELF,
    'CONTINENTAL_SLOPE': lambda cla: (cla==CONTINENTAL_SLOPE)|(cla==SCARPS),
    'CONTINENTAL_RISE': lambda cla: (cla==CONTINENTAL_RISE),
    'ABYSSAL_PLAIN': lambda cla: (cla==ABYSSAL_PLAIN),
    'BIG_ABP': lambda cla: (cla==ABYSSAL_PLAIN)|(cla==CONTINENTAL_RISE),
    'ROUGH_SEAFLOOR': lambda cla: (cla==ROUGH_SEAFLOOR)|(cla==VERY_ROUGH_SEAFLOOR),
    'MID_OCEANIC_RIDGE': lambda cla: (cla==MID_OCEANIC_RIDGE),
}

def _clm():
    n_cm= [None]*10
    n_cm[ABYSSAL_PLAIN]='blue'
    n_cm[CONTINENTAL_SHELF]='yellow'
    n_cm[CONTINENTAL_RISE]='lawngreen'
    n_cm[CONTINENTAL_SLOPE]='orange'
    n_cm[SCARPS]='m'
    n_cm[ROUGH_SEAFLOOR]="dodgerblue"
    n_cm[MID_OCEANIC_RIDGE]='r'
    n_cm[VERY_ROUGH_SEAFLOOR]='aqua'
    n_cm[LAND]='black'
    n_cm[SEAMOUNT]='white'
    mcm=c.LinearSegmentedColormap.from_list('class_color_map',n_cm)
    mcm.set_bad('white')
    return mcm
class_color_map=_clm()


def _clm_samples():
    n_cm= [None]*11
    n_cm[0]='blue'
    n_cm[1]='m'
    n_cm[2]='lawngreen'
    n_cm[3]='yellow'
    n_cm[4]='black'
    n_cm[5]="r"
    n_cm[6]='orange'
    n_cm[7]='dodgerblue'
    n_cm[8]='white'
    n_cm[9]='white'
    n_cm[10]='aqua'
    mcm=c.LinearSegmentedColormap.from_list('class_color_map',n_cm)
    mcm.set_bad('white')
    return mcm

class_color_map=_clm()
samples_color_map=_clm_samples()

filtre=np.zeros((21,21))
for i in range(21):
    for j in range(21):
        if (i-10)**2+(j-10)**2<=100:
            filtre[i,j]=1

def surface_texture(data,unit):
    tresh_flat=2
    vois=op.neighbor_grid(data)
    moy=np.median(vois,axis=(-1,-2))
    vois_big=op.neighbor_grid(np.abs(data-moy)>tresh_flat, 10)
    surf_text=100*np.mean((vois_big==1)*(filtre==1),axis=(-1,-2))/np.mean(filtre)
    return surf_text

def calc_slope(data,unit):
    the_map=Map(data,x_unit=unit,y_unit=unit)
    slope=np.arctan(the_map.norme_grad)*180/np.pi
    return slope

def surface_convexity(data,unit):
    tresh_fconc=0.0005
    the_map=Map(data,x_unit=unit,y_unit=unit)
    vois_big=op.neighbor_grid(the_map.laplacien<-tresh_fconc,10)
    conv_map=100*np.mean((vois_big==1)*(filtre==1), axis=(-1,-2))/np.mean(filtre)
    return conv_map

def nested_means_segment(slop,texture,conv, eight_classes=False,filtre=None):
    categs=-np.ones(slop.shape)

    #slope, slop_mean, slop_la_mean, slop_lq_mean
    def cut_offs(slop):
        slop_mean=slop.mean()
        slop_la=slop[slop<=np.median(slop)]
        slop_la_mean=slop_la.mean()
        slop_lq_mean=slop_la[slop_la<=np.median(slop_la)].mean()
        return slop_mean,slop_la_mean,slop_lq_mean
    
    if filtre is None:
        slop_mean,slop_la_mean,slop_lq_mean=cut_offs(slop)
        conv_mean, conv_la_mean, conv_lq_mean=cut_offs(conv)
        texture_mean, texture_la_mean, texture_lq_mean=cut_offs(texture)
    else:
        slop_mean,slop_la_mean,slop_lq_mean=cut_offs(slop[filtre])
        conv_mean, conv_la_mean, conv_lq_mean=cut_offs(conv[filtre])
        texture_mean, texture_la_mean, texture_lq_mean=cut_offs(texture[filtre])
        
        
    def questions(slop_cut,conv_cut,texture_cut):
        quest1=(slop>=slop_cut)
        quest2=(conv>=conv_cut)
        quest3=(texture>=texture_cut)
        return quest1,quest2,quest3
    
    quest1,quest2,quest3=questions(slop_mean,conv_mean,texture_mean)
    quest4,quest5,quest6=questions(slop_la_mean,conv_la_mean,texture_la_mean)
    quest7,quest8,quest9=questions(slop_lq_mean,conv_lq_mean,texture_lq_mean)
    

    sq11=quest1&quest2
    sq12=quest1&(~quest2)

    categs[sq11&quest3]=0
    categs[sq11&(~quest3)]=1
    categs[sq12&quest3]=2
    categs[sq12&(~quest3)]=3
    if eight_classes:
        sq8c1=(~quest1)&(quest2)
        sq8c2=(~quest1)&(~quest2)
        categs[sq8c1&quest3]=4
        categs[sq8c1&(~quest3)]=5
        categs[sq8c2&quest3]=6
        categs[sq8c2&(~quest3)]=7
    else:
        sq2=(~quest1)&quest4
        sq21=sq2&quest5
        sq22=sq2&(~quest5)

        categs[sq21&quest6]=4
        categs[sq21&(~quest6)]=5
        categs[sq22&quest6]=6
        categs[sq22&(~quest6)]=7

        sq3=(~quest1)&(~quest4)&quest7
        sq31=sq3&quest8
        sq32=sq3&(~quest8)

        categs[sq31&quest9]=8
        categs[sq31&(~quest9)]=9
        categs[sq32&quest9]=10
        categs[sq32&(~quest9)]=11

        sq4=(~quest1)&(~quest4)&(~quest7)
        sq41=sq4&quest8
        sq42=sq4&(~quest8)

        categs[sq41&quest9]=12
        categs[sq41&(~quest9)]=13
        categs[sq42&quest9]=14
        categs[sq42&(~quest9)]=15
    
    if not (filtre is None):
        categs[~filtre]=-1
    return categs

def full_segmentation(dem):
    """Segments the entire DEM with the nested means feature classification"""
    slope=op.divide_treatment(dem.elevation,lambda x: calc_slope(x,dem.unit))
    conv_map=op.divide_treatment(dem.elevation,lambda x: surface_convexity(x,dem.unit))
    surf_text=op.divide_treatment(dem.elevation,lambda x: surface_texture(x,dem.unit))
    above=(dem.elevation>=0)
    map_up=nested_means_segment(slope,surf_text,conv_map,above)
    map_oc=nested_means_segment(slope,surf_text,conv_map,~above)
    map_oc[map_oc==-1]=map_up[map_oc==-1]+16
    return map_oc

def complete_classification(dem):
    """Nested means is used on earth and the multivariate based classification is used for the ocean"""
    basic_slope=op.divide_treatment(dem.elevation,lambda x: calc_slope(x,dem.unit))
    conv_map=op.divide_treatment(dem.elevation,lambda x: surface_convexity(x,dem.unit))
    surf_text=op.divide_treatment(dem.elevation,lambda x: surface_texture(x,dem.unit))
    above=(dem.elevation>=0)
    map_up=nested_means_segment(basic_slope,surf_text,conv_map,above,eight_classes=True)
    sea_class=get_sea_classification(dem.slope,dem.roughness,dem.organization,dem.elevation,dem.curvature)
    sea_class[dem.elevation>=0]=16-map_up[dem.elevation>=0]
    return sea_class

def default_classification():
    return np.load('data/classes.npy')

def default_earth_indicators():
    slope=np.load('data/base_slope.npy')
    conv_map=np.load('data/conv_map.npy')
    surf_text=np.load('data/surf_text.npy')
    slope=block_reduce(slope,(2,2),np.mean)
    surf_text=block_reduce(surf_text,(2,2),np.mean)
    conv_map=block_reduce(conv_map,(2,2),np.mean)
    slope=slope[25:-25,:]
    surf_text=surf_text[25:-25,:]
    conv_map=conv_map[25:-25,:]
    return slope, surf_text, conv_map

def default_dem():
    dem=DEM.from_file('data/st.npy',curvature_path='data/curvature.npy',
                        organization_path='data/organization.npy',slope_path='data/slope.npy',
                        roughness_path='data/roughness.npy')
    dem.elevation=dem.elevation[250:-250,:]
    dem.curvature=dem.curvature[250:-250,:]
    dem.roughness=dem.roughness[250:-250,:]
    dem.organization=dem.organization[250:-250,:]
    dem.slope=dem.slope[250:-250,:]
    return dem