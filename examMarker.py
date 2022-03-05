import cv2 as cv
import numpy as np
import imutils
#from matplotlib import pyplot as plt
import easyocr
import json

#cv.imshow = lambda comment, image: None
#cv.waitKey = lambda key: None

file = open('outputs','w')
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

def getThreshold(img):
    # 1
    # img *= 255
    #img=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if np.max(img) <= 1:
        img = img*255
    img = img.astype('uint8')
    # 2
    H = np.histogram(img, list(range(0, 256)), 'dtype')[0]

    # 3
    sum_gray_levels = sum(grayLevel * value for grayLevel,
                          value in enumerate(H))
    sum_pixels = np.cumsum(H)[-1]
    T_init = round(sum_gray_levels/sum_pixels)

    T_old = T_init
    T_new = -100

    while(abs(T_old - T_new) > 0.001):
        if T_new == -100:
            T_new = T_old
        else:
            T_old = T_new
        # 4
        average_lower = 0.5
        count_lower = 0.5
        for grayLevel in range(0, int(T_old)):
            average_lower += grayLevel * H[grayLevel]
            count_lower += H[grayLevel]
        average_lower = round(average_lower / count_lower)

        average_higher = 0.5
        count_higher = 0.5
        for grayLevel in range(int(T_old), len(H)):
            average_higher += grayLevel * H[grayLevel]
            count_higher += H[grayLevel]

        average_higher = round(average_higher / count_higher)

        # 5
        T_new = (average_lower + average_higher)/2

    newImg = img.copy()
    mask0 = newImg <= T_new
    mask1 = newImg > T_new
    newImg[mask0] = 0
    newImg[mask1] = 1
    return newImg


def recursiveThreshold(oldImg, minS):
    height, width = oldImg.shape[0:2]

    img00 = oldImg[0:height//2, 0:width//2]
    img01 = oldImg[0:height//2, width//2:width]
    img10 = oldImg[height//2:height, 0:width//2]
    img11 = oldImg[height//2:height, width//2:width]

    if height < minS or width < minS:
        img00 = getThreshold(img00)
        img01 = getThreshold(img01)
        img10 = getThreshold(img10)
        img11 = getThreshold(img11)
    else:
        img00 = recursiveThreshold(img00, minS)
        img01 = recursiveThreshold(img01, minS)
        img10 = recursiveThreshold(img10, minS)
        img11 = recursiveThreshold(img11, minS)

    localImg = np.concatenate(
        (np.concatenate((img00, img10)), np.concatenate((img01, img11))), axis=1)
    return localImg

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


def biggestContour(contours, maxArea):
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


def getId (idImageBinary,idImage):
    idImageBinary= np.array(idImageBinary)
    idImage= np.array(idImage)
    id = ''
    cnts,hei = cv.findContours(idImageBinary, cv.RETR_EXTERNAL,	cv.CHAIN_APPROX_SIMPLE)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    ## name , id
    letters = []
    print('cnts : ',len(cnts))
    for c in cnts:
        (x, y, w, h) = cv.boundingRect(c)
        asceptRatio = w / float(h)
        #cv.rectangle(idImage,(x,y),(x+w,y+h),(0,255,0),1)
        print(w,h,asceptRatio)
        if 0<asceptRatio<2 and 5<w<40 and 5<h<40:
            character = idImageBinary[y:y+h,x:x+w]
            #entities.append(rect) 
            blank = character
        
            w,h = w+max(h,w),h+max(h,w)
            blank = np.pad(blank,((h,h), (w,w)),constant_values=0)
            ratio = 10
            print(h,w)
            #cv.imshow("blank",cv.resize(blank,(ratio*w,ratio*h)) )
            #cv.waitKey(0)
      
            result = reader.readtext(cv.resize(blank,(ratio*w,ratio*h)),detail=0)
            print(result)
            if  result :
                if result[0].isnumeric():
                    id += result[0]
    return id

def getName (nameBinary,name):
    # cv.imshow("nameBinary",nameBinary )
    # cv.waitKey(0)    
    result = reader.readtext(nameBinary,detail=0)
    return result[0]



def removeNumberLine(image,imageTh):
    SE = np.ones((80,5))
    # dilation
    numberDilation = cv.dilate(th4, SE, iterations=1)
    cv.imshow('vertical dilation',numberDilation)
    cv.waitKey(0)
    cnts,hei = cv.findContours(numberDilation, cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)

    cnts = sort_contours(cnts, method="left-to-right")[0]
    sripts = []
    for c in cnts:
        (x, y, w, h) = cv.boundingRect(c)
        if h >= image.shape[0] // 2: 
            sripts.append(c)
            #rect = cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
            #cv.imshow('rects',rect)
            #cv.waitKey(0)
    (x, y, w, h) =  cv.boundingRect(sripts[0])
    cv.imshow('columns',imageTh[:,x+w:])
    cv.waitKey(0)
    return imageTh[:,x+w:],image[:,x+w:]

def checkAnswers(row):
    factor = 9
    valid = True
    max_ = max(row)
    index = row.index(max_)
    for i in range(len(row)):
        if i!= index: 
            if factor+row[i] >= max_:
                valid = False

    return valid,index 



   
 ######################### start program #######################

def removeBorders(image):
    removed_h = int((image.shape[0]*0.05)//2)
    removed_w = int((image.shape[1]*0.08)//2)
    print('rem_h',removed_h,'rem_w',removed_w)
    image[:,:removed_w] = 0
    image[:,image.shape[1]-removed_w:] = 0
    image[:removed_h,:] = 0
    image[image.shape[0]-removed_h:,:] = 0
    return image

def BorderZeros(image):
    image[image.shape[0]-1,:]=0
    image[0,:]=0
    image[:,image.shape[1]-1]=0
    image[:,0]=0
    return image


def ExtractIdandName(headerBinary,header):

    cnts,hei = cv.findContours(headerBinary, cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE)

    ## sort contours by area
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:2]

    ## sort contours from left to right to ensure that name comes before id
    cnts = sort_contours(cnts, method="left-to-right")[0]

    ## name , id
    entities = []
    for c in cnts:
        (x, y, w, h) = cv.boundingRect(c)
        asceptRatio = w / float(h)
        cv.rectangle(header,(x,y),(x+w,y+h),(0,255,0),1)
        print(w,h,asceptRatio)
        if  w>20 and h >15:
            f=5
            rectBinary = headerBinary[min(y+f,headerBinary.shape[0]):min(y+h-f,headerBinary.shape[0]),max(x+f,0):max(x+w-f,0)]
            rect =  header[min(y+f,headerBinary.shape[0]):min(y+h-f,headerBinary.shape[0]),max(x+f,0):max(x+w-f,0)]
            entities.append((rectBinary,rect)) 
            cv.imshow("Bigs", rect)
            cv.waitKey(0)
            cv.destroyAllWindows()
    return entities

output = {}
modelAnswer = ['B']*50
models = []
Choices = ['A','B','C','D','E']
images = ["imgs/9.jpeg","imgs/10.jpeg","imgs/11.jpg","imgs/12.jpeg","imgs/13.jpeg","imgs/14.jpeg","imgs/15.jpeg","imgs/17.jpeg","imgs/20.jpeg","imgs/21.jpeg","imgs/22.jpeg"]
## take image
image = cv.imread(images[-1])
orignalImage = image.copy()

orignalImage = BorderZeros(orignalImage)
# get width and height
heigh = orignalImage.shape[0]
width = orignalImage.shape[1]

# convert image into grey scale
grayImage = cv.cvtColor(orignalImage, cv.COLOR_BGR2GRAY)

isScannered = False





# # get histogram
dst = cv.calcHist(grayImage, [0], None, [256], [0, 256])
# plt.hist(grayImage.ravel(), 256, [0, 256])
# plt.title('Histogram for gray scale image')
# plt.show()

# make gaussian blur
blur = cv.GaussianBlur(grayImage, (5, 5), 0)
##TODO:




# m7tagen n7dd el threshold @zizo
med_val = np.median(blur)
lower = int(max(0, 0.7*med_val))
upper = int(min(255, 1.3*med_val))
edged = cv.Canny(blur, lower, upper)
# cv.imshow("Outline", edged)
# cv.waitKey(0)



"""
save all found contours in contours
"""
contours, hirerarchy = cv.findContours(
    edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

## the paper is the biggest contour in image
biggest, maxArea = biggestContour(contours, width*heigh//5)

# biggest are 4 points of our biggest contour
if biggest.size != 0:
    # reorder these 4 points
    biggest = reorder(biggest)
    cv.drawContours(image, biggest, -1, (0, 255, 0), 20)
    imgBigContour = drawRectangle(image, biggest, 2)
    cv.imshow("paper outline", cv.resize(image, (600, 800)))
    cv.waitKey(0)
   

    # reorder sort points according to our annotations
    # extract exampaper from  image (warpPerspective)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width, 0], [0, heigh], [width, heigh]])
    paperView = cv.getPerspectiveTransform(pts1, pts2)
    wrap = cv.warpPerspective(orignalImage, paperView, (width, heigh))
    cv.imshow("get wrap", cv.resize(wrap, (600, 800)))
    cv.waitKey(0)
    cv.destroyAllWindows()




if biggest != []:
    examPaper = wrap
    orignalExampaper = wrap.copy()
    examPaper = cv.cvtColor(examPaper, cv.COLOR_BGR2GRAY)

else:
    orignalExampaper = orignalImage
    examPaper = grayImage


# threshold of exam paper
examPaper = recursiveThreshold(examPaper, 50)*255


cv.imshow("exampaper", cv.resize(examPaper, (600, 800)))
cv.waitKey(0)
# Inverte Image color
examPaper = 255 - examPaper
# remove noise from borders
thresh1 = removeBorders(examPaper)


cv.imshow("after threshold", cv.resize(thresh1, (600, 800)))
cv.waitKey(0)
cv.destroyAllWindows()


## sripts dilation 
for i in range(9,100,1):
    SE = np.ones((i, width))
    sriptedImage = cv.dilate(thresh1, SE, iterations=1)
    U,D = get_y_dim(sriptedImage)
    if D - U > heigh*0.65:
        break
    cv.imshow("stripted", sriptedImage)
    cv.waitKey(0)
    cv.destroyAllWindows()


upper, lower = get_y_dim(sriptedImage)
## get the header of the image which contains id and name
headerBinary = thresh1[:upper,:]
header = orignalExampaper[:upper,:]

## get the area contains bubbles
thresh1 = thresh1[upper:lower, :]
striptedImage = orignalExampaper[upper:lower, :]

cv.imshow("stripted", thresh1)
cv.waitKey(0)

cv.imshow("idImage", headerBinary)
cv.waitKey(0)
cv.destroyAllWindows()

######################  get id ##################################


entities = ExtractIdandName (headerBinary,header)
nameBinary = entities[0][0]
name = entities[0][1]
idBinary = entities[1][0]
id = entities[1][1]


id = getId(idBinary,id)
name = getName(nameBinary,name)


print('student name : ',name)
print('student id = ',id)
output['name'] = name
output['id'] = id

########################################################



## dilate the bubbles to make each group as contour
SE = np.ones((25, 30))
# dilation
dilation = cv.dilate(thresh1, SE, iterations=1)
cv.imshow("dilation", dilation)
cv.waitKey(0)
cv.destroyAllWindows()

## get contours of each bubbles group
contours, hirerarchy = cv.findContours(
    dilation.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# sort contours by area
contours = sorted(contours, key=cv.contourArea, reverse=True)

BubblesGroup = []

for con in contours:
    # approximate the contour
    peri = cv.arcLength(con, True)
    if cv.contourArea(con) < (striptedImage.shape[0]*striptedImage.shape[1]) // 15:
        break
    approx = cv.approxPolyDP(con, 0.02 * peri, True)
    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
        BubblesGroup.append(approx)

## sort contours from left to right
## to ensure that bubbles comes in correct order
BubblesGroup = sort_contours(BubblesGroup, method="left-to-right")[0]


wraps = []

iteration = 1
score = 0 
for subBubbles in BubblesGroup:

    subBubbles = reorder(subBubbles)
    pts1 = np.float32(subBubbles)
    pts2 = np.float32([[0, 0], [striptedImage.shape[1], 0], [0, striptedImage.shape[0]], [striptedImage.shape[1], striptedImage.shape[0]]])
    paperView = cv.getPerspectiveTransform(pts1, pts2)
    wrap = cv.warpPerspective(striptedImage, paperView, (striptedImage.shape[1], striptedImage.shape[0]))
    wraps.append(wrap)
    cv.imshow("final", wrap)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Start marking the Bubbles SubGroup
  
    wrapGrey =cv.cvtColor( wrap , cv.COLOR_BGR2GRAY)
    
    
    ret4, th4 = cv.threshold(wrapGrey, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    #th4 = recursiveThreshold(wrapGrey,wrapGrey.40)*255
    
    cv.imshow("number line th4", cv.resize(th4,(600,800)))
    cv.waitKey(0)
    
    # remove number from image
    imageTh,wrapBubbles = removeNumberLine(wrap,th4)
    cv.imshow('view',wrapBubbles)
    cv.waitKey(0)
    # convert image to grey scale
    wrapGreyBubbles =cv.cvtColor( wrapBubbles , cv.COLOR_BGR2GRAY)
    

    model = wrapBubbles.copy()
    ## get the bubbles
    cnts = cv.findContours(imageTh.copy(), cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print(len(cnts))
    questionCnts = []

    for bubble in cnts:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv.boundingRect(bubble)
        asceptRatio = w / float(h)
        #print(w, h, asceptRatio)
        ## only get the small bubbles 
        if w >= 10  and h >= 10 :
                questionCnts.append(bubble)
                image = cv.drawContours(wrapBubbles, bubble, -1, (0, 255, 0), 3)

    #draw bubbles
    cv.imshow("Get bubbles",image)
    cv.waitKey(0)



    ## sort contours from tom to bottom
    questionCnts = sort_contours(questionCnts, method="top-to-bottom")[0]

    correctAnwers = 0
    chooses = ['A','B','C','D','E']

    #thresh = th4
    thresh=  cv.threshold(wrapGreyBubbles, 0, 255,
        cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

 
    # each question has 5 possible answers, to loop over the
    # question in batches of 5
    for (q, i) in enumerate(np.arange(0, len(questionCnts), len(chooses))):

        # bubbled answer
            # sort each row form left to right 
            cnts = sort_contours(questionCnts[i:i +  len(chooses)],method="left-to-right")[0]
            # loop over the sorted contours
            row = []
            for (j, bubble) in enumerate(cnts):
                        mask = np.zeros(thresh.shape, dtype="uint8")
                        cv.drawContours(mask, [bubble], -1, 255, -1)
                        # count the number of non-zero pixels in the
                        # bubble area
                        mask = cv.bitwise_and(thresh, thresh, mask=mask)
                        (x, y, w, h) = cv.boundingRect(bubble)
                        print('x: ',x,'y : ',y,'w :',w,'h :',h)
                        cv.imshow("Get bubbles",mask)
                        cv.waitKey(0)
                        ## factor
                        f = 5
                        rect = mask[max(y-f,0):min(y+h+f,mask.shape[0]),max(x-f,0):min(x+w+f,mask.shape[1])]
                        cv.imshow("Get rect",rect)
                        cv.waitKey(0)
                        ## count the number of white pixels 
                        number_of_white_pix = np.sum(rect == 255)
                        percentage = (number_of_white_pix /(rect.shape[0]*rect.shape[1])) * 100
                        row.append(percentage)



            print('row = ',row)
            
            valid,choice = checkAnswers(row)
            if valid:
                if Choices[choice] == modelAnswer[iteration-1]:
                    #set color to green
                    color = (0, 255, 0)
                    print('corrent')
                    score += 1
                else:
                    #set color to red
                    color = (0, 0, 255)
                    print('wrong')
                cv.drawContours(model, [cnts[choice]], -1, color, 3)
                output['q'+str(iteration)] = Choices[choice]
            else:
                color = (255,0,0)
                cv.drawContours(model, cnts, -1, color, 3)
                print('invalid')
                output['q'+str(iteration)] = 'in valid answer'
    
            iteration += 1
    models.append(model)

output['score'] = score
file.write(json.dumps(output))

for m in models:
    cv.imshow('Student Answers',m)
    cv.waitKey(0)

file.close()

