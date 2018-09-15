#version python 3.7
#version opencv 3.4.2
#version Pillow 5.2.0

import cv2
import gzip
import numpy as np
import random
#import matplotlib
#matplotlib.use('TkAgg')
#from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
from mnist import *

#Class ImageFilter List 
##BLUR
##CONTOUR
##DETAIL
##EDGE_ENHANCE
##EDGE_ENHANCE_MORE
##EMBOSS
##FIND_EDGES
##SHARPEN
##SMOOTH
##SMOOTH_MORE

#Class ImageEnhance List 
##Color
##Contrast
##Brightness
##Sharpness

pillowImageFilter = { "GaussianBlur":ImageFilter.GaussianBlur,"BoxBlur":ImageFilter.BoxBlur,
                      "MaxFilter":ImageFilter.MaxFilter,"MinFilter":ImageFilter.MinFilter }
pillowImageEnhance = { "Contrast":ImageEnhance.Contrast,"Brightness":ImageEnhance.Brightness,"Sharpness":ImageEnhance.Sharpness }

class dataAugmentation:
    def Filter(images, index, digits, filterName, parameters):
        filterImage = []
        for i in range(0, len(images)):
            filtertype = pillowImageFilter[filterName]
            im = Image.fromarray(np.array(images[i], dtype = np.uint8))
            imf = im.filter(filtertype(parameters))
            #im.show("Org")
            #imf.show("Proc %f" % factor)  
            title = "./PNGs/Filter/"+ str(digits[i]) + "_" + str(index[i]) + "_" + filterName + "_" +str(parameters) + ".png"
            imf.save(title)
            filterImage.append(imf)
        return filterImage

    def Enhance(images, index, digits, enhancerName, parameters):
        enhanceImage = []
        for i in range(0, len(images)):         
            im = Image.fromarray(np.array(images[i], dtype = np.uint8))
            enhancer = pillowImageEnhance[enhancerName](im)         
            factor = 1 / parameters
            ime = enhancer.enhance(factor)
            #im.show("Org")
            #ime.show("Proc %f" % factor)           
            title = "./PNGs/Enhance/"+ str(digits[i]) + "_" + str(index[i]) + "_"  + enhancerName + "_" +str(parameters) + ".png"
            ime.save(title)
            enhanceImage.append(ime)
        return enhanceImage


def generate_augmentation_numbers_sequence(digits, spacingRange, imageWidth):
    
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
    return images ,index, digits

procOption = {"Filter":dataAugmentation.Filter,"Enhance":dataAugmentation.Enhance}

def data_augmentation(option, subOption, parameters):
    digits = [2, 0, 1, 8, 0, 9, 0, 7]
    spacingRange = (20, 30)
    imageWidth = 1024
    [images,index,digits] = generate_augmentation_numbers_sequence(digits, spacingRange, imageWidth)
    procImage = procOption[option](images, index, digits, subOption, parameters) 
    return procImage


