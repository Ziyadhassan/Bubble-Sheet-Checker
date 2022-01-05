from commonfunctions import show_images
import cv2 as cv
import typing
import numpy as np
import imutils
from matplotlib import pyplot as plt
from utils import ptile,ring_se
from skimage.filters import threshold_local
import utils
import functools
import easyocr

def get_y_dim(stripped_image):
    j = stripped_image.shape[1] // 2
    flips = [0]
    diff, lower, upper = -1, -1, -1
    prv_clr = stripped_image[0][j]
    for i in range(stripped_image.shape[0]):
        if prv_clr != stripped_image[i][j]:
            prv_clr = stripped_image[i][j]
            flips.append(i)
            if flips[-1] - flips[-2] > diff:
                diff  = flips[-1] - flips[-2]
                upper = flips[-2]
                lower = flips[-1]

    return (upper, lower)

def get_largest_cols(stripped_image):
    i = stripped_image.shape[0] // 2
    # The first flip point.
    flips = [0]
    prv_clr = stripped_image[i][0]
    # Loop over the image width and record every line of color change.
    for j in range(stripped_image.shape[1]):
        if prv_clr != stripped_image[i][j]:
            prv_clr = stripped_image[i][j]
            flips.append(j)
    # The last flip point.
    flips.append(stripped_image.shape[1])
    # Get the column sizes we have.
    diffs = [j2 - j1 for j1, j2 in zip(flips, flips[1:])]
    max_dif = max(diffs)
    # Get the indices of all the columns that have
    # comparable sizes to the max column size.
    indices = [index for index, value in enumerate(diffs)
               # If some column has the max column size -5%
               # we may include it.
               if (max_dif * 0.95 < value)]
    # Return the indicies of the columns we found.
    return [(flips[index], flips[index + 1]) for index in indices]


def biggestContour(contours, maxArea=5000):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv.contourArea(i)
        # to avoid small contours (noise)
        if area > maxArea:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri, True)
            # check if this rectangle
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def drawRectangle(img, biggest, thickness):
    cv.line(img, (biggest[0][0][0], biggest[0][0][1]),
            (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[0][0][0], biggest[0][0][1]),
            (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[3][0][0], biggest[3][0][1]),
            (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[3][0][0], biggest[3][0][1]),
            (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)

    return img


def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew

def sort_contours(cnts, method="top-to-bottom"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)


def get_nth_peak(array: typing.List, n, x):
    # Remove consequitive duplicates.
    unq_array = [array[0]] + \
                [val1 for val1, val2 in zip(array[1:], array[:-1])
                 if val1 != val2]
    # Pad the array with a min value so that the first and 
    # last elements in the array can be selected.
    min_val = min(unq_array) - 1
    unq_array.insert(0, min_val)
    unq_array.append(min_val)
    # Get the peaks.
    peaks = [val for left, val, right
             in zip(unq_array[:-2], unq_array[1:-1], unq_array[2:])
             if (left+x < val) and (val > right+x)]
    return array.index(peaks[n - 1])

######################### start program #######################
images = ["imgs/5.jpg","imgs/11.jpg","imgs/10.jpeg"]
## take image
image = cv.imread(images[2])
orignalImage = image.copy()
## no need for resize
# orignalImage = cv.resize(image,(800,800)) #resizing because opencv does not work well with bigger images

# get width and height
heigh = orignalImage.shape[0]
width = orignalImage.shape[1]

# convert image into grey scale
grayImage = cv.cvtColor(orignalImage, cv.COLOR_BGR2GRAY)

# get histogram
dst = cv.calcHist(grayImage, [0], None, [256], [0, 256])
plt.hist(grayImage.ravel(), 256, [0, 256])
plt.title('Histogram for gray scale image')
plt.show()

# make gaussian blur
blur = cv.GaussianBlur(grayImage, (5, 5), 0)
ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
print(ret3)
cv.imshow('thresh', th3)
cv.waitKey(0)

"""
# structure element
SE =np.ones((5,5),np.uint8)
# opening
openingImage = cv.morphologyEx(blur, cv.MORPH_OPEN, SE)
# followed by closing with same SE
closingImage = cv.morphologyEx(blur, cv.MORPH_CLOSE, SE)
"""

# m7tagen n7dd el threshold @zizo
med_val = np.median(blur)
lower = int(max(0, 0.7*med_val))
upper = int(min(255, 1.3*med_val))
edged = cv.Canny(blur, lower, upper)
cv.imshow("Outline", edged)
cv.waitKey(0)

## msh 3arf lsa btboz leh??
# structure element
# SE = np.ones((5,5))
# Dilation = cv.dilate(edged,SE,iterations = 2)
# erosion = cv.erode(edged,SE,iterations = 1)


"""
save all found contours in contours
"""
contours, hirerarchy = cv.findContours(
    edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
## the paper is the biggest contour in image
biggest, maxArea = biggestContour(contours, width*heigh//5)

# biggest are 4 points of our biggest contour
print(biggest)
if biggest.size != 0:
    # reorder these 4 points
    biggest = reorder(biggest)
    cv.drawContours(image, biggest, -1, (0, 255, 0), 20)
    imgBigContour = drawRectangle(image, biggest, 2)
    cv.imshow("Big Contours", cv.resize(image, (600, 800)))
    cv.waitKey(0)
    # reorder sort points according to our annotations
    # extract exampaper from  image (warpPerspective)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width, 0], [0, heigh], [width, heigh]])
    paperView = cv.getPerspectiveTransform(pts1, pts2)
    wrap = cv.warpPerspective(orignalImage, paperView, (width, heigh))
    cv.imshow("final", cv.resize(wrap, (600, 800)))
    cv.waitKey(0)

# if their is no biggest contour (contour with large size)
#   then the image is already the exam paper so take the greyImage
if biggest != []:
    examPaper = wrap
    examPaper = cv.cvtColor(examPaper, cv.COLOR_BGR2GRAY)
else:
    examPaper = grayImage


## histogram of the examPaper itself
cv.imshow("bubble group", cv.resize(examPaper, (600, 800)))
cv.waitKey(0)
plt.hist(examPaper.ravel(), 256, [0, 256])
plt.title('Histogram for gray scale image')
plt.show()


"""
####   hsv thresholding   (tare2a 7lwa mmkn nst5dmha)
imgHsv = cv.cvtColor(examPaper, cv.COLOR_BGR2HSV)

# Define lower/upper color
lower = np.array([0, 0, 180])
upper = np.array([180, 20, 255])

# Check the region of the image actually with a color in the range defined below
# inRange returns a matrix in black and white
bw = cv.inRange(imgHsv, lower, upper)
print(bw)
"""


#TODO: 3ayzen nzbt el thresholding
ret, thresh1 = cv.threshold(examPaper, 220, 255, cv.THRESH_BINARY_INV)
#th = utils.ptile(examPaper, 100-85)
#thresh1 = np.where(examPaper > th, 255, 0)
# show_images([thresh1])
cv.imshow("threshold", thresh1)
cv.waitKey(0)




# contours, hirerarchy = cv.findContours(
#     dilation.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#TODO: for 2nd model  (bta3 mostafa gendy nzwd showyt hyperParameter)
# SE = np.ones((25, 25))
# # dilation
# thresh1 = cv.dilate(thresh1, SE, iterations=1)
# cv.imshow("dilation", thresh1)
# cv.waitKey(0)

## sripts dilation 
SE = np.ones((25, width))
stripped_image = cv.dilate(thresh1, SE, iterations=1)
cv.imshow("stripted", stripped_image)
cv.waitKey(0)

## get the sripts and (biggest sripts is the bubbles group)
upper, lower = get_y_dim(stripped_image)
thresh1 = thresh1[upper:lower, :]
striptedImage = orignalImage[upper:lower, :]
cv.imshow("stripted", thresh1)
cv.waitKey(0)

SE = np.ones((thresh1.shape[0],10))
stripped_image = cv.dilate(thresh1, SE, iterations=1)
cv.imshow("stripted ver", stripped_image)
cv.waitKey(0)
col_indices = get_largest_cols(stripped_image)
wraps = [striptedImage[:, col_index[0]:col_index[1]] for col_index in col_indices]
for part in wraps:
    cv.imshow("hi",cv.resize(part,(600,800)))
    cv.waitKey(0)

'''
## dilate the bubbles to make each group as contour
SE = np.ones((20, 20))
# dilation
dilation = cv.dilate(thresh1, SE, iterations=1)
cv.imshow("dilation", dilation)
cv.waitKey(0)

## get contours of each bubbles group
contours, hirerarchy = cv.findContours(
    dilation.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# sort contours by area
contours = sorted(contours, key=cv.contourArea, reverse=True)

BubblesGroup = []

for con in contours:
    # approximate the contour
    peri = cv.arcLength(con, True)
    if cv.contourArea(con) < (width*heigh) // 8:
        break
    approx = cv.approxPolyDP(con, 0.02 * peri, True)
    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
        BubblesGroup.append(approx)
BubblesGroup = sort_contours(BubblesGroup, method="left-to-right")[0]
print(len(BubblesGroup))
'''

# wraps = []
# for subBubbles in BubblesGroup:

#     subBubbles = reorder(subBubbles)
#     #cv.drawContours(striptedImage, subBubbles, -1, (0, 255, 0), 20)
#     #imgBigContour = drawRectangle(striptedImage, subBubbles, 2)
#     #cv.imshow("Big Contours", cv.resize(striptedImage, (600, 800)))
#     #cv.imshow()
#     #cv.waitKey(0)
#     pts1 = np.float32(subBubbles)
#     pts2 = np.float32([[0, 0], [width, 0], [0, heigh], [width, heigh]])
#     paperView = cv.getPerspectiveTransform(pts1, pts2)
#     wrap = cv.warpPerspective(striptedImage, paperView, (width, heigh))
#     wraps.append(wrap)
#     cv.imshow("final", wrap)
#     cv.waitKey(0)

# تصحيييييح
wrapGrey =cv.cvtColor( wraps[0] , cv.COLOR_BGR2GRAY)
## 
ret4, th4 = cv.threshold(wrapGrey, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
# find contours in the thresholded image, then initialize
# the list of contours that correspond to questions
cv.imshow('before circle dilate',th4)
cv.waitKey(0)

###
noOfWhites = []
for i in range(0,100,1):
    s1 = cv.erode(th4,ring_se(i,i+1))
    whitePixels = np.sum(s1 == 255)
    noOfWhites.append(whitePixels)

#######
x = 10
########
bubbleRadius = get_nth_peak(noOfWhites,2,x)
print(noOfWhites,bubbleRadius)
bubble_centers = cv.erode(th4,ring_se(bubbleRadius,bubbleRadius+1))
cv.imshow("omar : ",bubble_centers)
cv.waitKey(0)


#th4 = th4[:,]


### tare2a atl3 beha el bubbles bs manf3tsh
"""
# Initialize parameter settiing using cv2.SimpleBlobDetector
params = cv.SimpleBlobDetector_Params()
# Set Area filtering parameters
params.filterByArea = True
params.minArea = 20
# Set Circularity filtering parameters
params.filterByCircularity = False
params.minCircularity = 0.9
 
# Set Convexity filtering parameters
params.filterByConvexity = False
params.minConvexity = 0.2
     
# Set inertia filtering parameters
params.filterByInertia = False
params.minInertiaRatio = 0.01



# Create a detector with the parameters
detector = cv.SimpleBlobDetector_create(params)
	
# Detect blobs
keypoints = detector.detect(th4)

# Draw blobs on our image as red circles
blank = np.zeros((1, 1))
blobs = cv.drawKeypoints(th4, keypoints, blank, (0, 0, 255),
                          cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
print('number_of_blobs = ',number_of_blobs)
# Show blobs
cv.imshow("Filtering Circular Blobs Only", blobs)
cv.waitKey(0)
cv.destroyAllWindows()

"""




## get the bubbles
cnts = cv.findContours(th4.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print(len(cnts))
questionCnts = []
# loop over the contours
cv.imshow("Big",wrapGrey)
cv.waitKey(0)
for bubble in cnts:
	# compute the bounding box of the contour, then use the
	# bounding box to derive the aspect ratio
	(x, y, w, h) = cv.boundingRect(bubble)
	asceptRatio = w / float(h)
	# in order to label the contour as a question, region
	# should be sufficiently wide, sufficiently tall, and
	# have an aspect ratio approximately equal to 1
	print(w, h, asceptRatio)
    ## only get the small bubbles 
	if w >= 10 and w<=50 and h >= 10 and h <= 50  and 0.90<asceptRatio<1.10  :
            # cv.drawContours(wrap[0], [c], -1, (0, 255, 0), 20)
            # imgBigContour = drawRectangle(wrap[0], c, 2)
            # cv.imshow("Big Contours",wrap[0])
            # cv.waitKey(0)
            #image = cv.rectange(wraps[0], (x +w//2,y+h//2), 10, (255, 0, 0), 1)
            #image = cv.rectangle(wraps[0],(x +w//2,y+h//2),(w,h),(0,255,0),2)
            questionCnts.append(bubble)
            # draw bubbles contour
            image = cv.drawContours(wraps[0], bubble, -1, (0, 255, 0), 3)
            #cv.imshow("Get bubbles",image)
            #cv.waitKey(0)

#print(questionCnts)
cv.imshow("Get bubbles",image)
cv.waitKey(0)



## sort contours from tom to bottom
questionCnts = sort_contours(questionCnts, method="top-to-bottom")[0]

correctAnwers = 0
chooses = ['A','B','C','D','E']

thresh = cv.threshold(wrapGrey, 0, 255,
	cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
# each question has 5 possible answers, to loop over the
# question in batches of 5
iteration = 1
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):

	# bubbled answer
        # sort each row form left to right 
        cnts = sort_contours(questionCnts[i:i + 5],method="left-to-right")[0]
        bubbled = None
        # loop over the sorted contours
        row = []
        for (j, bubble) in enumerate(cnts):
                    #print(' : ' , bubble)
                
            # construct a mask that reveals only the current
            # "bubble" for the question
                    mask = np.zeros(thresh.shape, dtype="uint8")
                    cv.drawContours(mask, [bubble], -1, 255, -1)
                    # apply the mask to the thresholded image, then
                    # count the number of non-zero pixels in the
                    # bubble area
                    mask = cv.bitwise_and(thresh, thresh, mask=mask)
                    (x, y, w, h) = cv.boundingRect(bubble)
                    print('x: ',x,'y : ',y,'w :',w,'h :',h)
                    #rect = cv.rectangle(mask,(x,y),(x+w,y+h),255,2)
                    cv.imshow("Get bubbles",mask)
                    cv.waitKey(0)
                    ## factor
                    f = 0
                    rect = mask[max(y-f,0):min(y+h+f,mask.shape[0]),max(x-f,0):min(x+w+f,mask.shape[1])]
                    cv.imshow("Get rect",cv.resize(rect,(600,800)))
                    cv.waitKey(0)
                    number_of_white_pix = np.sum(rect == 255)
                    percentage = (number_of_white_pix /(rect.shape[0]*rect.shape[1])) * 100
                    row.append(percentage)

                    # total = cv.countNonZero(mask)
                    # #print(total)
                    # if bubbled is None or total > bubbled[0]:
                    #     bubbled = (total, j)
                    #     print(bubbled)

        print('row = ',row)
        ##TODO: test bs
        answers=[] 
        max_ = max(row)

        for (k, r) in enumerate(row):
            if r > 50.0:
                answers.append(k)

        # only one chosen answer
        if len(answers) == 1:
            result = 'question {} : {} \n'.format(iteration,chooses[answers[0]])
        else:
            result = 'question {} : {} \n'.format(iteration,'kaaaaaak')
        file.write(result)
        iteration += 1




file.close()