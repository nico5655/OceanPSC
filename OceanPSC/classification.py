import matplotlib.colors as c
import numpy as np
import operations as op
from Map import Map
import cv2

#everything below tresh is of class
#slope
flat_tresh=0.9#RadToDeg*np.arctan(1/100)
sloping_tresh=1.9#RadToDeg*np.arctan(1/32)
steep_tresh=3.6
#roughness
smooth_thresh=0.20
rough_tresh=0.48
#organization
disorganized_treshold=0.75
organized_treshold=0.85
#elevation
abyssal_tresh=-4600
deep_tresh=-3600
intermediate_tresh=-1600
#convexity
#we separate by (+) and (-)

#identifiers
ABYSSAL_PLAIN=3
VERY_ROUGH_SEAFLOOR=1
ROUGH_SEAFLOOR=2
CONTINENTAL_SHELF=7
UPPER_CONTINENTAL_SLOPE=6
LOWER_CONTINENTAL_SLOPE=5
CONTINENTAL_RISE=4
MID_OCEANIC_RIDGE=0
SCARPS=8


def clean(cls_map):
    rslt=cls_map.copy()
    vois=op.neighbor_grid(rslt,4,bound_method="duplicate")
    max_decider=np.zeros((rslt.shape))
    arg_max_decider=rslt.copy()
    for k in range(9):
        decider=(vois==k).sum(axis=(-1,-2))
        arg_max_decider[decider>max_decider]=k
        arg_max_decider[decider==max_decider]=rslt[decider==max_decider]
        max_decider[decider>max_decider]=decider[decider>max_decider]
    return arg_max_decider
    
def get_sea_classification(slope,roughness,organization,elevation,curvature):
    ## terrain classification
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
    ## terrain regions
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

    cos=np.float32(regions.copy())
    cos[elevation>=0]=0
    cos=clean(clean(cos))
    cos=clean(clean(cos))
    cos=clean(clean(cos))
    ## classes
    #auxiliary variables
    abyssal=elevation<abyssal_tresh
    deep=(elevation<deep_tresh)&(~abyssal)
    intermediate=(elevation<intermediate_tresh)&(~(abyssal|deep))
    shallow=~(abyssal|intermediate|deep)

    convex=curvature>0
    concave=~convex
    #the 5th category is "land" and we use another classification for this one


    SMOOTH=(regions==0)
    ORGANIZED=(regions==1)
    SLOPING=(regions==2)
    SCARPED=(regions==3)
    ROUGH=(regions==4)
    VERY_ROUGH=(regions==5)
    convex=curvature>0
    concave=~convex
    abyssal=elevation<abyssal_tresh
    deep=(elevation<deep_tresh)&(~abyssal)
    intermediate=(elevation<intermediate_tresh)&(~(abyssal|deep))
    shallow=~(abyssal|intermediate|deep)

    Abyssal_Plain=SMOOTH&(intermediate|deep|abyssal)
    Continental_Shelf=(SMOOTH|ORGANIZED|ROUGH)&shallow
    Continental_Rise=ORGANIZED&(intermediate|deep|abyssal)
    Upper_Continental_Slope=SLOPING&convex
    Lower_Continental_Slope=SLOPING&concave
    Scarps=SCARPED
    Rough_Seafloor=ROUGH&(deep|abyssal)
    Mid_Oceanic_Ridge=(ROUGH&intermediate)|(VERY_ROUGH&(shallow|intermediate))
    Very_Rough_Seafloor=VERY_ROUGH&(deep|abyssal)

    #result
    final_classes=np.zeros(elevation.shape)
    final_classes[Abyssal_Plain]=ABYSSAL_PLAIN
    final_classes[Continental_Shelf]=CONTINENTAL_SHELF
    final_classes[Continental_Rise]=CONTINENTAL_RISE
    final_classes[Upper_Continental_Slope]=UPPER_CONTINENTAL_SLOPE
    final_classes[Lower_Continental_Slope]=LOWER_CONTINENTAL_SLOPE
    final_classes[Scarps]=SCARPS
    final_classes[Rough_Seafloor]=ROUGH_SEAFLOOR
    final_classes[Mid_Oceanic_Ridge]=MID_OCEANIC_RIDGE
    final_classes[Very_Rough_Seafloor]=VERY_ROUGH_SEAFLOOR
    #correction
    final_classes[(elevation>-100)&Mid_Oceanic_Ridge]=CONTINENTAL_SHELF
    
    #result primary cleaning
    old=final_classes.copy()
    final_classes[elevation>=0]=CONTINENTAL_SHELF
    final_classes=np.ma.masked_where(final_classes==SCARPS,final_classes)
    final_classes=np.uint8(final_classes)
    final_classes=divide_treatment(final_classes,lambda dtr: np.ma.median(op.neighbor_grid(np.uint8(dtr),4),axis=(-1,-2)),r=2)
    
    final_classes=np.array(final_classes)
    final_classes[old==SCARPS]=SCARPS
    final_classes=np.int8(final_classes)
    final_classes[((final_classes==VERY_ROUGH_SEAFLOOR)|(final_classes==ROUGH_SEAFLOOR))&intermediate]=MID_OCEANIC_RIDGE
    final_classes=clean(final_classes)
    final_classes=clean(final_classes)
    
    #result advanced cleaning
    def close_class(clas_c,main_class,crushed_classes,iters=10):
        clas_c=clas_c.copy()
        img=np.float32(clas_c==main_class)
        kernel = np.ones((5,5),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iters)
        predicat=(clas_c==crushed_classes[0])
        for clas in crushed_classes[1:]:
            predicat=predicat|(clas_c==clas)
        clas_c[(img==1)&predicat]=main_class
        return clas_c
    def dilate_adjacency_protection(clas_c,main_class,adjacency,crushed_classes):
        clas_c=clas_c.copy()
        img=np.float32(clas_c==main_class)
        kernel = np.zeros((5,5),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, iterations=17)
        predicat=(clas_c==crushed_classes[0])
        for clas in crushed_classes[1:]:
            predicat=predicat|(clas_c==clas)
        clas_c[(img==1)&predicat]=adjacency
        return clas_c
    
    #key closings (eliminating scattered classes)
    final_classes=close_class(final_classes,MID_OCEANIC_RIDGE,
                       [CONTINENTAL_RISE,CONTINENTAL_SHELF,UPPER_CONTINENTAL_SLOPE,LOWER_CONTINENTAL_SLOPE])

    final_classes=close_class(final_classes,CONTINENTAL_SHELF,[MID_OCEANIC_RIDGE,CONTINENTAL_RISE])
    final_classes=close_class(final_classes,LOWER_CONTINENTAL_SLOPE,[MID_OCEANIC_RIDGE,CONTINENTAL_RISE])
    final_classes=close_class(final_classes,UPPER_CONTINENTAL_SLOPE,[MID_OCEANIC_RIDGE,CONTINENTAL_RISE])
    final_classes=dilate_adjacency_protection(final_classes,CONTINENTAL_SHELF,UPPER_CONTINENTAL_SLOPE,[MID_OCEANIC_RIDGE])
    
    #less important, half esthetic closings
    final_classes=close_class(final_classes,CONTINENTAL_RISE,[ROUGH_SEAFLOOR,VERY_ROUGH_SEAFLOOR],iters=3)
    final_classes=close_class(final_classes,UPPER_CONTINENTAL_SLOPE,[ROUGH_SEAFLOOR,VERY_ROUGH_SEAFLOOR],iters=4)
    final_classes=close_class(final_classes,LOWER_CONTINENTAL_SLOPE,[ROUGH_SEAFLOOR,VERY_ROUGH_SEAFLOOR],iters=4)
    final_classes=close_class(final_classes,VERY_ROUGH_SEAFLOOR,[CONTINENTAL_RISE,UPPER_CONTINENTAL_SLOPE,LOWER_CONTINENTAL_SLOPE],iters=2)
    final_classes=close_class(final_classes,ROUGH_SEAFLOOR,[CONTINENTAL_RISE,UPPER_CONTINENTAL_SLOPE,LOWER_CONTINENTAL_SLOPE],iters=2)
    final_classes=close_class(final_classes,VERY_ROUGH_SEAFLOOR,[CONTINENTAL_RISE,ROUGH_SEAFLOOR],iters=3)
    final_classes=close_class(final_classes,ABYSSAL_PLAIN,[ROUGH_SEAFLOOR,VERY_ROUGH_SEAFLOOR])
    final_classes=close_class(final_classes,ABYSSAL_PLAIN,[CONTINENTAL_RISE],iters=2)
    final_classes=close_class(final_classes,CONTINENTAL_RISE,[ABYSSAL_PLAIN],iters=2)
    final_classes=close_class(final_classes,ROUGH_SEAFLOOR,[ABYSSAL_PLAIN],iters=1)
    final_classes=close_class(final_classes,MID_OCEANIC_RIDGE,[ABYSSAL_PLAIN],iters=2)

    final_classes[elevation>=0]=np.nan
    return np.float32(final_classes)


def class_col_map():
    n_cm= [None]*9
    n_cm[ABYSSAL_PLAIN]='blue'
    n_cm[CONTINENTAL_SHELF]='yellow'
    n_cm[CONTINENTAL_RISE]='lawngreen'
    n_cm[UPPER_CONTINENTAL_SLOPE]='orange'
    n_cm[LOWER_CONTINENTAL_SLOPE]='tab:orange'
    n_cm[SCARPS]='m'
    n_cm[ROUGH_SEAFLOOR]="dodgerblue"
    n_cm[MID_OCEANIC_RIDGE]='r'
    n_cm[VERY_ROUGH_SEAFLOOR]='aqua'
    mcm=c.ListedColormap(n_cm)
    mcm.set_bad('white')
    return mcm
class_color_map=class_col_map()


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
    the_map=Map(dem.elevation,x_unit=unit,y_unit=unit)
    vois_big=op.neighbor_grid(the_map.laplacien<-tresh_fconc,10)
    conv_map=100*np.mean((vois_big==1)*(filtre==1), axis=(-1,-2))/np.mean(filtre)
    return conv_map

def nested_means_segment(slop,texture,conv,filtre=None):
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
    surf_text=surface_texture(dem)
    slope=calc_slope(dem)
    conv_map=surface_convexity(dem)
    above=(dem.elevation>=0)
    map_up=nested_means_segment(slope,surf_text,conv_map,above)
    map_oc=nested_means_segment(slope,surf_text,conv_map,~above)
    map_oc[map_oc==-1]=map_up[map_oc==-1]+16
    return map_oc

def complete_classification(dem):
    """Nested means is used on earth and the multivariate based classification is used for the ocean"""
    basic_slope=op.divide_treatment(dem.elevation,calc_slope)
    conv_map=divide_treatment(dem.elevation,surface_convexity)
    surf_text=divide_treatment(dem.elevation,surface_texture)
    above=(dem.elevation>=0)
    map_up=nested_means_segment(basic_slope,surf_text,conv_map,above)*(9/16)
    sea_class=get_sea_classification(dem.slope,dem.roughness,dem.organization,dem.elevation,dem.curvature)
    sea_class[dem.elevation>=0]=map_up[dem.elevation>=0]
    return sea_class