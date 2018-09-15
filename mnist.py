#version python 3.7
#version opencv 3.4.2
#version Pillow 5.2.0

import cv2
import gzip
import numpy as np
import random

digitHeight = 28
digitWidth = 28

def read_labels_from_file(filename): # For reading a label file
    labels = {0: [],1: [],2: [],3: [],4: [],5: [],6: [],7: [],8: [],9: []}
    # Using "with" for files limits the scope of variables to within the block. No file closing means no forgetting to close = good.
    with gzip.open(filename, "rb") as f:
        # Pointer at 0, read the magic number - 801 means 1 dimention, 802 means 2 dim. etc etc
        magic = f.read(4) # First 4 bytes = magic number
        magic = int.from_bytes(magic, "big") # Convert bytes to int format, 'big' = big-endian / most significant byte first
        print("Magic: ", magic)

        # Pointer now at 3 (4th pos)
        numLabels = f.read(4) # Next 4 bytes = number of items in the file
        numLabels = int.from_bytes(numLabels, "big")

        print("Num of labels: ", numLabels)

        # Fill the table 
        for c in range(numLabels):
           labels[int.from_bytes(f.read(1),"big")].append(c) # For every column add the current byte (pixel) to the column array
    return labels

def read_images_from_file(filename,digits): # For reading an image file (not the same structure as a label file)
    with gzip.open(filename, "rb") as f:
        # Same as label function, read magic number
        magic = f.read(4)
        magic = int.from_bytes(magic, "big")
        print("Magic: ", magic)

        # Next 4 bytes = number of items (images)
        numImgs = f.read(4)
        numImgs = int.from_bytes(numImgs, "big")
        print("Num of Images: ", numImgs)

        # Next 4 = number of rows
        numRows = f.read(4)
        numRows = int.from_bytes(numRows, "big")
        print("Num of Rows: ", numRows)

        # Next 4 = number of columns
        numCols = f.read(4)
        numCols = int.from_bytes(numCols, "big")
        print("Num of Cols: ", numCols)

        images = []
        # read all the images listed in the digits
        for i in digits:
            f.seek(numRows * numCols * i + 16,0)
            rows = []
            for r in range(numRows):
                cols = []
                for c in range(numCols):
                    pixel =   int.from_bytes(f.read(1), "big")
                    #pixel = pixel / 255.0
                    cols.append(pixel) # For every column add the current byte (pixel) to the column array
                rows.append(cols) # For every row, add the column array
            images.append(rows) # For every image, add the rows array - rows + columns of pixels = full image.
        
        #images = images*255.0 # convert [0,255] to [0,1]
    return images #Return the image array

def save_images(images, digits, start, spacing, imageWidth):
   
    #set a blank buffer for padding images
    blankImage = np.ones((digitHeight,imageWidth,1),dtype = np.float32)

    step = start #for recordin the last (x + width) coordination of digit image

    for i in range(0,len(digits)):

        digitImage =  np.array(images[i], dtype = np.float32)
        interval = spacing + digitWidth
        # calculate every (x + width) from start to end
        if i > 0:
            step +=  interval

        #make a  mask for every digit
        mask = np.zeros(blankImage.shape, dtype = np.float32)
        mask[0:digitHeight, 0:digitWidth] = True
        locs = np.where(mask != 0) # Get the non-zero mask locations
        
        #copy every digit to blank image next to the last digit
        blankImage[locs[0], locs[1] + step] = digitImage[locs[0], locs[1], None]

    cv2.imwrite("digits_data.png",blankImage)
    return blankImage

def calculate_distribution(digits, spacingRange, imageWidth):

    distribute = []
    lowerRange = spacingRange[0]
    upperRange = spacingRange[1]
    l = len(digits)
    default = []

    #judge distribution of all the digit in the image
    #image width is too small to contian all of the digits
    condition = imageWidth - digitWidth * l
    if condition < 1:
        print("imageWidth should not less than digitWidth*len(digits)! \n")
        return

    spacing = 0
    start = imageWidth - spacing * (l - 1) - digitWidth * l
    start = np.int(np.floor(start / 2))
    default.append(start)
    default.append(spacing)
    
    #min spacing < max spacing
    condition = (lowerRange - upperRange)
    
    if condition >= 0:
        print("should set spacingRange[0]< spacingRange[1]!" + \
            "return default distribution! ")
        return default

    #no minus value
    condition = lowerRange * upperRange
    if condition < 0:
        print("please use positive integer in spacingRange!" + \
            "return default distribution! ")
        return default

    #min possible spacing = (imgWidth - spacing*(l - 1) - digitWidth*l)/2
    #condition = np.int((imageWidth - spacingRange[0]*(len(digits) - 1) - digitWidth*len(digits))/2)
    
    condition = imageWidth - lowerRange * (l - 1) - digitWidth * l
    condition = np.int(condition / 2)
    if condition < 1:
        print("spacingRange is too larger to fit the imageWidth!" + \
            "return default distribution! ")
        return default

    #calculate a proper max spacing (too larger is no meaning) 
    maxSpacing = 0 #if only 1 digit maxSpacing is 0
    if (l - 1) > 0:#inputed more than 1 digit 
        #maxSpacing = np.int((imageWidth - digitWidth * l)/(l - 1))
        maxSpacing = imageWidth - digitWidth * l
        maxSpacing = np.int(maxSpacing / (l - 1))
        
    condition = (maxSpacing - upperRange) * (maxSpacing - lowerRange)
    if condition <= 0:
        print("set spacingRange to a proper value!" + \
            "return default distribution! ")
        return default 

    if l == 1:
        lowerRange = 0
        upperRange = 1

    #find the spacing randomly then calculate start coordinate for the first digit according to image width
    start = -1
    while start < 0 :#find a "start" >= 0
        #print("lower=",lowerRange,"upper=",upperRange)
        spacing = random.randrange(lowerRange, upperRange, 2)
        start = imageWidth - spacing * (l - 1) - digitWidth * l
        start = np.int(np.floor(start / 2))
    else: print("distribute = ", [start,spacing])
    
    distribute.append(start)
    distribute.append(spacing)
    
    return distribute

def generate_numbers_sequence(digits, spacingRange, imageWidth):
    
    print("\n[function]:calculate distribution")
    distribute = calculate_distribution(digits, spacingRange, imageWidth)
    
    if distribute is None:
        return

    start = distribute[0]
    spacing = distribute[1]

    print("\n[function]:read label file")
    labels = read_labels_from_file("data/train-labels-idx1-ubyte.gz")

    #generate random index
    index = []
    for i in digits:
        if i < 0 and i > 9:
            print("please input number in [0,9]!")
        j = random.randrange(0, len(labels[i]), 2)
        index.append(labels[i][j])

    print("digits index in mnist = ",index)
    print("\n[function]:read images")
    images = read_images_from_file('data/train-images-idx3-ubyte.gz',index)
    
    print("\n[function]:save images")
    saveImage = save_images(images, digits, start, spacing, imageWidth)
    return saveImage
             
def random_test():

    for i in range(0,10000):
        dnum = random.randrange(1, 10, 1)
        digits = []
        for d in range(0,dnum):
            digits.append(random.randrange(0, 10, 1))

        lower = random.randrange(0, 32, 1)
        upper = random.randrange(0, 64, 1)
        spacingRange = (lower, upper)
        imageWidth = random.randrange(512, 1024, 1)
        saveImage = generate_numbers_sequence(digits, spacingRange, imageWidth)

        img = saveImage * 255.0 #convert [0,1] to [0,255]
        cv2.imshow("digits_data", img) 
        cv2.waitKey(1000) 
        print("\ntest", i, "times!")



