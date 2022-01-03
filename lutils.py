from skimage.morphology.selem import disk
from commonfunctions import *

def conv2_255(img):
  '''Converts an image from any scale to 0-255 scale.'''
  gray_image = rgb2gray(img).astype(np.float64)
  im_max, im_min = gray_image.max(), gray_image.min()
  rng = im_max - im_min
  gray_image -= im_min
  gray_image *= 255
  gray_image /= rng
  return gray_image.astype(np.uint8)

def ptile(img, p=50):
  '''Get the p value at which p% of the histogram is on the left.'''
  img = conv2_255(img)
  his = histogram(img, 256)
  tot_pix = img.shape[0] * img.shape[1] * p / 100
  tot_cov = 0
  for npix, glvl in zip(his[0], his[1]):
    if tot_cov >= tot_pix:
      break
    tot_cov += npix
  return glvl

def ring_se(inner_rad, outer_rad):
  assert outer_rad > inner_rad
  disc = disk(outer_rad)
  disc[outer_rad-inner_rad:outer_rad+inner_rad+1,
       outer_rad-inner_rad:outer_rad+inner_rad+1] -= disk(inner_rad)
  return disc