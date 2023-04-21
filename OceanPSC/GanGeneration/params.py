nbr_entrainement = 100
bruit_dim = 100
embedding_dim = 100
nbr_exemples = 11
num_classes = 11
buffer_size = 100
LAMBDA = 10
drift = 0.001
maxi,mini = 5307.6294, -10760.56

dir_nbr = '.'
checkpoint_path = dir_nbr + "/train_checkpoints6/"
images_path=dir_nbr+'/Images7/'

class_names=['abp', 'active_margin', 'cr', 'csh', 'land_csh', 'mor', 'passive_margin', 'rs',
 's_abp', 's_rs', 'vrs']

import numpy as np

#mean/5000 and std/1000 defaults for each type
vals = np.array([[-0.96872544,  0.11005837],
       [-0.38071564,  1.4304543 ],
       [-0.710909  ,  0.3437929 ],
       [-0.02952676,  0.039938  ],
       [ 0.00547793,  0.14771624],
       [-0.60483205,  0.25452018],
       [-0.3528577 ,  0.82931316],
       [-0.89659214,  0.21191935],
       [-0.8248253 ,  0.88628983],
       [-0.8695726 ,  0.7519231 ],
       [-1.0097973 ,  0.26679662]], dtype=np.float32)
