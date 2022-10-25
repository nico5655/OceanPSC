import OceanPSC.operations as op
import OceanPSC.utils as utils
import numpy as np

class Map(object):
    """Representation of a map, a 2D field of scalar and additional informations"""
    def __init__(self, data, x_unit=1, y_unit=1):
        self.x_unit = x_unit
        self.y_unit = y_unit
        self.data = data

    @property
    def grad(self):
        return op.grad(self.data, x_dist=self.x_unit, y_dist=self.y_unit, clean_bounds=True)

    @property
    def norme_grad(self):
        return op.norme(self.grad)

    @property
    def laplacien(self):
        return op.laplacien(self.data,x_dist=self.x_unit,y_dist=self.y_unit)

    @property
    def dimensions(self):
        return (self.x_unit * self.data.shape[0],self.y_unit * self.data.shape[1])

    def from_file(path):
        data = utils.load_data(path)
        return Map(data)

    def indicator_grid(self,calc,transformation=lambda x: x.data, reduce=(54,54)):
        data_t=transformation(self)
        k,_,_ = op.create_indicator_grid(data_t,calc,reduce)
        return k

    def get_indicators_data(self,indicators, reduce=(54,54)):
        data=[]
        labels=[]

        ind_grid=[]
        for calc,trans in indicators:
            if trans != None:
                ind_grid.append(self.indicator_grid(calc,trans,reduce=reduce))
                continue
            ind_grid.append(self.indicator_grid(calc,reduce=reduce))

        for i in range(self.data.shape[0]//reduce[0]):
            for j in range(self.data.shape[1]//reduce[1]):
                a=[gr[i,j] for gr in ind_grid]
                data.append(a)
                labels.append((i,j))

        return data,labels

