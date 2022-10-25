from PIL import Image
import PIL
import skimage.measure
import numpy as np

PIL.Image.MAX_IMAGE_PIXELS = 466560010

def load_data(path,reduction=(4,4)):
    img=Image.open(path)
    rsl=np.array(img)
    rsl = skimage.measure.block_reduce(rsl, (4,4), np.mean)
    return rsl