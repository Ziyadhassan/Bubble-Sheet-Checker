import numpy as np
import cv2 as cv
import inspect

from skimage.color import rgb2gray
from skimage.exposure import histogram
from skimage.morphology.selem import disk


def imshow(img, wait_key=0):
    if not hasattr(imshow, '_count'):
        imshow._count = 0
    try:
        img_name = [var_name for var_name, var_val
                    in inspect.currentframe()
                    .f_back.f_locals.items()
                    if var_val is img][0]
    except:
        img_name = "Unnamed " + str(imshow._count)
        imshow._count += 1
    cv.imshow(img_name, img)
    cv.waitKey(wait_key)


def conv2_255(img):
    '''Converts an image from any scale to 0-255 scale and gray scales it.'''
    gray_image = rgb2gray(img).astype(np.float64)
    im_max, im_min = gray_image.max(), gray_image.min()
    im_rng = (im_max - im_min) or 1
    gray_image = (gray_image - im_min) * 255 / im_rng
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
    assert outer_rad > inner_rad, 'The inner radius must be less than outer radius.'
    disc = disk(outer_rad)
    disc[outer_rad-inner_rad:outer_rad+inner_rad+1,
         outer_rad-inner_rad:outer_rad+inner_rad+1] -= disk(inner_rad)
    return disc


def binary_thresh(img):
    img = conv2_255(img)
    hist = histogram(img, nbins=256)
    graylevels = hist[1]
    num_pixels = hist[0]
    th = round(np.sum(graylevels * num_pixels) / img.size)
    prev_th = not th
    while(th != prev_th):
        prev_th = th
        th_higher = round(np.sum(
            graylevels[graylevels > th] * num_pixels[graylevels > th]) /
            np.sum(num_pixels[graylevels > th])
        )
        th_lower = round(np.sum(
            graylevels[graylevels <= th] * num_pixels[graylevels <= th]) /
            np.sum(num_pixels[graylevels <= th])
        )
        th_higher = 0 if th_higher == np.nan else th_higher
        th_lower = 0 if th_lower == np.nan else th_lower
        th = (th_higher + th_lower) // 2
    thresh_image = np.where(img > th, 255, 0).astype(np.uint8)
    return thresh_image


def resize_factor(img, factor=1):
    h, w = img.shape[0], img.shape[1]
    return cv.resize(img, w * factor, h * factor)


def resize_hw(img, *, new_h=None, new_w=None):
    if (new_h and new_w) is not None:
        return cv.resize(img, new_w, new_h)
    elif new_h is not None:
        return resize_factor(img, new_h / img.shape[0])
    elif new_w is not None:
        return resize_factor(img, new_w / img.shape[1])
    else:
        raise RuntimeError('Give either width or height.')

