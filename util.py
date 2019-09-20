import imageio
import numpy as np
from tensorflow.python.client import device_lib

def save2img(d_img, fn):
    _min, _max = d_img.min(), d_img.max()
    if np.abs(_max - _min) < 1e-4:
        img = np.zeros(d_img.shape)
    else:
        img = (d_img - _min) * 255. / (_max - _min)
    
    img = img.astype('uint8')
    imageio.imwrite(fn, img)

def scale2uint8(d_img):
#     _min, _max = d_img.min(), d_img.max()
    np.nan_to_num(d_img, copy=False)
    _min, _max = np.percentile(d_img, 0.05), np.percentile(d_img, 99.95)
    d_img = d_img.clip(_min, _max)
    if _max == _min:
        d_img -= _max
    else:
        d_img = (d_img - _min) * 255. / (_max - _min)
    return d_img.astype('uint8')
