#version python 3.7
#version opencv 3.4.2
#version Pillow 5.2.0  
from mnist import *
from generate import *


##-------------------------------------------[Generate Digits]------------------------------------------------

########### Test Code Randomly ############
#random_test()
########### Test Code Randomly ############

digits = [2, 0, 1, 8, 0, 9, 0, 7]
spacingRange = (20, 30)
imageWidth = 1024
saveImage = generate_numbers_sequence(digits, spacingRange, imageWidth)

img = saveImage * 255.0 #convert [0,1] to [0,255]
cv2.imshow("digits_data", img)
cv2.imwrite("digits_image.png",img)   
cv2.waitKey(1000)

##-------------------------------------------[Data Augmentation]----------------------------------------------
print("-------------------[info]-------------------")
print("\n1.",end="")
print("Following image filter keyword can be used to select a filter:")
for key, value in pillowImageFilter.items() :
    print(key)

print("\n2.",end="")
print("Following image enhance keyword can be used to select a filter:")
for key, value in pillowImageEnhance.items() :
    print(key)

print("-------------------[info]-------------------")

#Select proper parameter for different option
# MaxFilter is Dilation
# MinFilter is Erosion
data_augmentation("Filter","MaxFilter",3)
data_augmentation("Enhance","Brightness",2)

