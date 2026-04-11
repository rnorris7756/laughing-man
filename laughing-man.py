#
# laughingMan.py
#
# Uses pyOpenCV for video capture, face detection, and video display.
# Uses PIL for creating the overlaid image and mask.
#
# Based on the face detection example in pyOpenCV and the Processing
# implementation of same functionality <http://www.awgh.org/?p=21>.
#
# Jouni Paulus, jouni.paulus(a)iki.fi, 16.5.2010

import sys
import pyopencv
from PIL import Image, ImageDraw, ImageChops

# face detection templates
cascadeName = "haarcascade_frontalface_alt.xml"

# two figures for the Laughing Man mask, first is a static one, second 
# the rotating text
stableImageName = "limg.png"
rotImageName = "ltext.png"

# don't try to detect too small faces
minFaceSize = pyopencv.Size(100, 100) 

# degrees, the caching will use 360/rotReso times the overlay image
# size the memory
rotReso = 5

# first order IIR low-pass for the ROI coordinates 
# newROI = roiLambda*prevROI + (1-roiLambda)*foundROI
roiLambda = 0.95
# if no face is found, the earlier ROI will be faded out
fadeoutLambda = 0.99
# ROI smaller than this will be regarded as empty
fadeoutLim = 50
# prevROI will be stored as floats. otherwise the rounding errors mess
# the LP filtering
prevROI = None 

# increase the actual mask from the detected face a bit
roiScaler = 1.3

keyWaitDelay = 25

mainWinName = "pyLaughingMan, v.0.1"

def detectAndDraw(img, cascade, overlayImg, maskImg):
    global prevROI

    # copy input image into processing input: grayscale, histogram equalised
    smallImg = pyopencv.Mat()
    pyopencv.cvtColor(img, smallImg, pyopencv.CV_BGR2GRAY)
    pyopencv.equalizeHist(smallImg, smallImg)

    # detect only the largest face -> huge speedup
    faces = cascade.detectMultiScale( smallImg,
        1.1, 2, 0
        |pyopencv.CascadeClassifier.SCALE_IMAGE
        |pyopencv.CascadeClassifier.FIND_BIGGEST_OBJECT,
        minFaceSize ).to_list_of_Rect()

    overlayImgRe = pyopencv.Mat()
    maskImgRe = pyopencv.Mat()
    imSize = img.size()
    if len(faces)==0:
        # no face detected. fade out existing ROI
        r = None
        if prevROI!=None:
            r = pyopencv.Rect()
            # slowly reduce the size of earlier ROI
            roiCX = prevROI[0] + prevROI[2]/2.0
            roiCY = prevROI[1] + prevROI[3]/2.0

            newWidth = fadeoutLambda*prevROI[2]
            newHeight = fadeoutLambda*prevROI[3]
            
            newX = roiCX - newWidth/2.0
            newY = roiCY - newHeight/2.0

            # plain type cast truncates
            r.x = int(round(newX))
            r.y = int(round(newY))
            r.width = int(round(newWidth))
            r.height = int(round(newHeight))
            prevROI = (newX, newY, newWidth, newHeight)
            if r.width<fadeoutLim or r.height<fadeoutLim:
                r = None
                prevROI = None
    else:
        # found a face. filter ROI coordinates and apply image modification
        r = faces[0]
        if prevROI==None:
            prevROI = (r.x*1.0, r.y*1.0, r.width*1.0, r.height*1.0)
        else:
            newWidth = roiLambda*prevROI[2] + (1.0-roiLambda)*r.width*roiScaler
            newHeight = roiLambda*prevROI[3] + (1.0-roiLambda)*r.height*roiScaler
            roiCX = roiLambda*(prevROI[0] + prevROI[2]/2.0) + (1.0-roiLambda)*(r.x + r.width/2.0)
            roiCY = roiLambda*(prevROI[1] + prevROI[3]/2.0) + (1.0-roiLambda)*(r.y + r.height/2.0)

            newX = roiCX - newWidth/2.0
            newY = roiCY - newHeight/2.0
  
            # cap the ROI borders to image borders
            if newX<0:
                newX = 0
            if newY<0:
                newY = 0
            if newX+newWidth>imSize[0]:
                newWidth = imSize[0] - newX
            if newY+newHeight>imSize[1]:
                newHeight = imSize[1] - newY

            prevROI = (newX, newY, newWidth, newHeight)

            # plain type cast truncates
            r.x = int(round(newX))
            r.y = int(round(newY))
            r.width = int(round(newWidth))
            r.height = int(round(newHeight))

            if r.width<fadeoutLim or r.height<fadeoutLim:
                r = None
                prevROI = None

    if r!=None:
        roiSize = r.size()
        faceROI = img(r) # set ROI

        # resize the mask and overlay image 
        overlayImgRe = pyopencv.Mat.from_pil_image(overlayImg.resize((roiSize[0], roiSize[1])))
        maskImgRe = pyopencv.Mat.from_pil_image(maskImg.resize((roiSize[0], roiSize[1])))
        
        # ugly hack to draw full overlay, but didn't find any other way to 
        # force replacement of the mask area
        pyopencv.subtract(faceROI, faceROI, faceROI, maskImgRe)
        pyopencv.add(faceROI, overlayImgRe, faceROI, maskImgRe)
        
    pyopencv.imshow(mainWinName, img)
    
if __name__ == '__main__':
    capture = pyopencv.VideoCapture()
    frame = pyopencv.Mat()
    
    cascade = pyopencv.CascadeClassifier()

    if not cascade.load( cascadeName ):
        print("ERROR: Could not load classifier cascade")
        sys.exit(-1)

    capture.open(0)

    pyopencv.namedWindow(mainWinName, pyopencv.CV_WINDOW_AUTOSIZE&1)

    stImg = Image.open(stableImageName)
    rotImg = Image.open(rotImageName)
    # for some bug, the Image.im data is not loaded in open for pngs
    stImg.load()
    rotImg.load()
    
    # R,G,B,A -> A band
    stBands = stImg.split()
    rotBands = rotImg.split()
    stAlpha = stBands[3]
    rotAlpha = rotBands[3]
    rotAngle = 0

    tmpOffset = 0.1
    imSz = stImg.size

    maskImg = Image.new('L', stImg.size)
    maskDraw = ImageDraw.Draw(maskImg)
    maskDraw.ellipse((imSz[0]*tmpOffset, imSz[1]*tmpOffset, imSz[0]*(1-tmpOffset), imSz[1]*(1-tmpOffset)), 'white')
    maskImg = ImageChops.lighter(ImageChops.lighter(maskImg, stAlpha), rotAlpha)

    if capture.isOpened():
        # this is just to optimise the memory consumption of the caching
        # do not store the original, quite large pics, but smaller versions
        inputHeight = capture.get(pyopencv.CV_CAP_PROP_FRAME_HEIGHT)
        inputWidth = capture.get(pyopencv.CV_CAP_PROP_FRAME_WIDTH)

        minDim = int(inputHeight)
        if inputWidth<inputHeight:
            minDim = int(inputWidth)

        maskImg = maskImg.resize((minDim, minDim))
        
        # create empty list
        imgCache = [[] for rIdx in range(360/rotReso)]

        print "Press ESC to quit."

        while True:
            capture.retrieve(frame)
            if frame.empty():
                break

            # since the basic image is the same whenever the rotation angle
            # is the same, trade some memory for speed-up by caching.
            # cache fill could be done before starting the other operations
            # so that the initial slowing wouldn't be noticed, but this way
            # operation starts immediately and it's possible to see the 
            # benefit from caching
            cacheIdx = rotAngle/rotReso
            if len(imgCache[cacheIdx])<1:
                print "Calculating the rotated image for angle %s." % rotAngle
                # create the image to be drawn on face
                # first, white circle background
                combImg = Image.new('RGB', stImg.size)    
                drawImg = ImageDraw.Draw(combImg)
                drawImg.ellipse((imSz[0]*tmpOffset, imSz[1]*tmpOffset, imSz[0]*(1-tmpOffset), imSz[1]*(1-tmpOffset)), 'white')
    
                # then the rotated text and static image with alpha-based masks
                tmpImg = Image.composite(stImg, Image.composite(rotImg.rotate(rotAngle, Image.NEAREST), combImg, rotAlpha.rotate(rotAngle, Image.NEAREST)), stAlpha)

                tmpImg = tmpImg.resize((minDim, minDim))
                imgCache[cacheIdx].append(tmpImg)
            else:
                # retrieve a cached image
                tmpImg = imgCache[cacheIdx][0]

            # find a face and draw the mask
            detectAndDraw(frame, cascade, tmpImg, maskImg)
            
            # rotate the text
            rotAngle = (rotAngle+rotReso)%360
            if pyopencv.waitKey(keyWaitDelay) >= 0:
                break

