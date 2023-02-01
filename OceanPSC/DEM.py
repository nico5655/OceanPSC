import numpy as np

## constants
default_unit=8*450.0
PLANNAR=0
PIT=1
CHANNEL=2
PASS=3
RIDGE=4
PEAK=5
RadToDeg=(180/np.pi)
Tslope_min=0.6
Tconvex=0.25

class DEM(object):
    """Representation of a digital elevation model (unit is the meters distance between two points)."""
    def __init__(self, data, unit=default_unit):
        self.unit = unit
        self.elevation = data
        self.__slope=None
        self.__organization=None
        self.__curvature=None
        self.__roughness=None


    @property
    def slope(self):
        if self.__slope ==None:
            raise Exception('Indicators haven\'t been initialized, you should call \'calc_all_indicators\' (it might take time).')
        return self.__slope
    @slope.setter
    def slope(self,val):
        if val.shape!=self.shape:
            raise ValueError('Shape {val.shape} doesn\'t match shape of DEM ({self.shape})')
        self.__slope=val
        
    @property
    def slope(self):
        if self.__slope ==None:
            raise Exception('Indicators haven\'t been initialized, you should call \'calc_all_indicators\' (it might take time, and you should use op.divide_treatment).')
        return self.__slope
    @slope.setter
    def slope(self,val):
        if val.shape!=self.shape:
            raise ValueError('Shape {val.shape} doesn\'t match shape of DEM ({self.shape})')
        self.__slope=val

    @property
    def roughness(self):
        if self.__roughness ==None:
            raise Exception('Indicators haven\'t been initialized, you should call \'calc_all_indicators\' (it might take time, and you should use op.divide_treatment).')
        return self.__roughness
    @roughness.setter
    def roughness(self,val):
        if val.shape!=self.shape:
            raise ValueError('Shape {val.shape} doesn\'t match shape of DEM ({self.shape})')
        self.__roughness=val

    @property
    def organization(self):
        if self.__organization ==None:
            raise Exception('Indicators haven\'t been initialized, you should call \'calc_all_indicators\' (it might take time, and you should use op.divide_treatment).')
        return self.__organization
    @organization.setter
    def organization(self,val):
        if val.shape!=self.shape:
            raise ValueError('Shape {val.shape} doesn\'t match shape of DEM ({self.shape})')
        return self.__slope
        self.__organization=val

    @property
    def curvature(self):
        if self.__curvature ==None:
            raise Exception('Indicators haven\'t been initialized, you should call \'calc_all_indicators\' (it might take time).')
        return self.__curvature
    @curvature.setter
    def curvature(self,val):
        if val.shape!=self.shape:
            raise ValueError('Shape {val.shape} doesn\'t match shape of DEM ({self.shape})')
        self.__curvature=val


    @property
    def shape(self):
        return elevation.shape

    @property
    def dimensions(self):
        """The DEM spatial dimensions (unit*shape)"""
        return (self.unit * self.shape[0],self.unit * self.shape[1])

    def from_file(path,slope_path=None,roughness_path=None,organization_path=None,curvature_path=None):
        """Create a DEM from file elevation data and eventually loads indicators/classification if files are provided"""
        data = np.load(path)
        dem=DEM(data)
        if slope_path != None:
            dem.slope=np.load_data(slope_path)
        if roughness_path != None:
            dem.roughness=np.load_data(roughness_path)
        if organization_path != None:
            dem.organization=np.load_data(organization_path)
        if curvature_path != None:
            dem.curvature=np.load_data(curvature_path)
        return dem


    def save(self,path):
        np.save(path / 'dem/elevation.npy',self.elevation)
        np.save(path / 'dem/slope.npy',self.slope)
        np.save(path / 'dem/roughness.npy',self.roughness)
        np.save(path / 'dem/organization.npy',self.organization)
        np.save(path / 'dem/curvature.npy',self.curvature)


def _classify(self,a,b,c,d,e,f,radius):
    """Run the classification on an area caracterized by the 6 coefficients,
    on the given radius (same radius as for the coeff calculations)"""
    rslt_class=np.zeros(self.shape)#unasigned will be PLANAR
    slope=np.arctan(np.sqrt(d**2+e**2))*RadToDeg
    n=2*radius+1
    #cross-sec convexity, set to 0 by convention when d=e=0, ie slope=0, but actually not used in that case
    crosc=-20*n*g*(b*d*d + a*e*e - c*d*e)/(d**2 + e**2+(d==0)*(e==0))
    maxic =20*n*g*(-(a+b)+np.sqrt((a-b)**2 + c**2))
    minic = -20*n * g * (a+b+np.sqrt((a-b)**2 + c**2))
    #max/min conv
    #used when slope=0, or near, which means crosc can't be calculated (denominator 0)
    #convexity on approximately flat surfaces
    #case 1: sloping
    high_slope=(slope>Tslope_min)
    ridge_pred1=high_slope&(crosc>Tconvex)
    rslt_class[ridge_pred1]=RIDGE
    channel_pred1=high_slope&(crosc<-Tconvex)
    rslt_class[channel_pred1]=CHANNEL
    
    #case 2: horizontal surface
    low_slope=~high_slope
    low_slope_and_hconv=low_slope&(maxic>Tconvex)
    peak_pred=low_slope_and_hconv&(minic>Tconvex)
    rslt_class[peak_pred]=PEAK
    pass_pred2=low_slope_and_hconv&(minic<-Tconvex)
    ridge_pred2=low_slope_and_hconv&(~pass_pred2)
    rslt_class[pass_pred2]=PASS
    rslt_class[ridge_pred2]=RIDGE
    low_slope_and_lconv=low_slope&(minic<-Tconvex)
    pit_predicate=low_slope_and_lconv&(maxic<-Tconvex)
    channel_predicate2=low_slope_and_lconv&(~pit_predicate)
    rslt_class[pit_predicate]=PIT
    rslt_class[channel_predicate2]=CHANNEL
    return rslt_class

def classify_area(self,radius):
    """Run classify on the coeffs calculated for this area."""
    a,b,c,d,e,f=calc_coeffs(study_area,radius,self.unit)
    return classify(a,b,c,d,e,f,radius,self.unit)


def calculate_all_indicators(self,r_range):
    """Calculate all the 4 indicators to be used for main classification
    (3 primary geometric signatures and 1 auxilliary variable)
    (technically there are 5 indicators, but the 5th is the elevation itself)
    All indicators all calculated simultaneously since they all require the save heavy coefficients calculation.
    """
    #slope
    slop_rs=np.zeros(([len(r_range)]+list(study_area.shape)))
    #aspect is slope direction
    aspects_rs=np.zeros(([len(r_range)]+list(study_area.shape)))
    #surface curvature
    conv_rs=np.zeros(([len(r_range)]+list(study_area.shape)))
    #ps will contain for each class, and each point, the proportion of the class across the r_range
    #ie, (number of detections of this class at this point across the range)/len(r_range)
    ps=np.zeros([6]+list(study_area.shape))
    Em=-np.log(1/6)#6 classes (the maximum entropy used to normalize E)
    cpt=0
    for r in r_range:
        a,b,c,d,e,f=calc_coeffs(study_area,r,self.unit)
        slop_rs[cpt]=np.arctan(np.sqrt(d**2+e**2))*RadToDeg
        conv_rs[cpt]=-200*(2*r+1)*self.unit*(a*d*d + b*e*e + c*e*d)/((((d==0)*(e==0))+d**2 + e**2)*(1 +d**2+e**2)**(3/2))
        aspects_rs[cpt]=np.arctan(e/d)*RadToDeg
        #classification
        clasr=classify(a,b,c,d,e,f,r,self.unit)
        for i in range(5):#from PLANNAR to PIT
            ps[i]+=(clasr==i)
        cpt+=1
    ps=ps/len(r_range)
    E=-np.sum(ps*np.log(ps+(ps==0)),axis=0)#entropy
    avg_slope=np.mean(slop_rs,axis=0)
    #the more the slope direction is dispersed, the more disorganized the terrain is
    #if the slope direction is the same at all scales, it means all the terrain folows the same trend (for exemple a big straight downwoard slope)
    #but if it changes a lot based on scale, that means the small scale and large scale changes aren't the same
    #(a generally decreasing, but locally increasing slope for example)
    organization=np.nanstd(aspects_rs,axis=0)
    organization=np.nanmax(organization)-organization
    #avg_curvature is an auxiliary variable
    avg_curvature=np.mean(conv_rs,axis=0)

    #roughness is normalized entropy
    #if the features detected are consistant on all scales, that means our terrain is smooth
    #but, if depending on your zoom you see a mountain or a pit, that means the terrain is rough
    self.roughness=E/Em
    self.slope=avg_slope
    self.organization=organization
    self.curvature=avg_curvature