# import cv2
# from skimage.util.dtype import img_as_bool, img_as_ubyte
# from commonfunctions import *


# def myHistogram(gray_image):
#     hist = np.zeros(257)
#     for i in gray_image:
#         for j in i:
#             hist[j] += 1
#     return hist


# def get_choices(img):
#     img = rgb2gray(img)
#     img /= img.max()
#     img = img_as_ubyte(img)
#     showHist(myHistogram(img))

#     return {'Q1': 'C', 'Q6': 'A', 'Q3': 'F'}

# def conv2_255(gray_image):
#   '''Converts an image from any scale to 0-255 scale.'''
#   im_max, im_min = gray_image.max(), gray_image.min()
#   rng = im_max - im_min
#   gray_image -= im_min
#   gray_image /= rng
#   gray_image *= 255
#   return gray_image.astype(np.uint8)


# img = cv2.imread('imgs/8.png')
# #choices = get_choices(img)
# print(conv2_255(rgb2gray(img)))

from skimage.morphology import opening, closing, diameter_closing, disk
from commonfunctions import *
import lutils

def get_choices(img):
    # Scale the image in 0-255 scale then negate it.
    img = 255 - lutils.conv2_255(img)
    show_images([img])
    
    img = median(img)
    
    show_images([img])
    
    #th = (int(img.min()) + img.max()) // 2
    th = lutils.ptile(img=img)
    print(th)
    
    img[img <= th] = 0
    img[img > th] = 1
    dia_SE = disk(5)
    
    show_images([img])
    #binary_dilation
    #binary_erosion
    #dilation(erosion(img, dia_SE), dia_SE)
    img = closing(img, np.ones((9, 6)))
    return img

    return {'Q1': 'C', 'Q6': 'A', 'Q3': 'F'}


img = io.imread('imgs/8.png')
choices = get_choices(img)
show_images([img, choices])