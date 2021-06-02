import os
import sys
import cv2
import numpy as np
#import imgaug  
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
#from skimage import io
from matplotlib import pyplot as plt
from PIL import Image

from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

def checkImage(image):
    """
    Args:
        image: input image to be checked
    Returns:
        binary image
    Raises:
        RGB image, grayscale image, all-black, and all-white image

    """
    if len(image.shape) > 2:
        print("ERROR: non-binary image (RGB)"); sys.exit();

    smallest = image.min(axis=0).min(axis=0) # lowest pixel value: 0 (black)
    largest  = image.max(axis=0).max(axis=0) # highest pixel value: 1 (white)

    if (smallest == 0 and largest == 0):
        print("ERROR: non-binary image (all black)"); sys.exit()
    elif (smallest == 255 and largest == 255):
        print("ERROR: non-binary image (all white)"); sys.exit()
    elif (smallest > 0 or largest < 255 ):
        print("ERROR: non-binary image (grayscale)"); sys.exit()
    else:
        return True

class Toolbox:
    def __init__(self, image):
        self.image = image

    @property
    def printImage(self):
        """
        Print image into a file for checking purpose
        unitTest = Toolbox(image);
        unitTest.printImage(image);
        """
        f = open("image_results.dat", "w+")
        for i in range(0, self.image.shape[0]):
            for j in range(0, self.image.shape[1]):
                f.write("%d " %self.image[i,j])
            f.write("\n")
        f.close()
        
    @property
    def displayImage(self):
        """
        Display the image on a window
        Press any key to exit
        """
        cv2.imshow('Displayed Image', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def saveImage(self, title, extension):
        """
        Save as a specific image format (bmp, png, or jpeg)
        """
        cv2.imwrite("{}.{}".format(title,extension), self.image)

    def morph_open(self, image, kernel):
        """
        Remove all white noises or speckles outside images
        Need to tune the kernel size
        Instruction:
        unit01 = Toolbox(image);
        kernel = np.ones( (9,9), np.uint8 );
        morph  = unit01.morph_open(input_image, kernel);
        """
        bin_open = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)
        return bin_open

    def morph_close(self, image, kernel):
        """
        Remove all black noises or speckles inside images
        Need to tune the kernel size
        Instruction:
        unit01 = Toolbox(image);
        kernel = np.ones( (11,11)_, np.uint8 );
        morph  = unit01.morph_close(input_image, kernel);
        """        
        bin_close = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
        return bin_close


def trimap(image, size1,size2):
    """
    This function creates a trimap based on simple dilation algorithm
    Inputs [4]: a binary image (black & white only), name of the image, dilation pixels
                the last argument is optional; i.e., how many iterations will the image get eroded
    Output    : a trimap
    """
    checkImage(image)
    row    = image.shape[0]
    col    = image.shape[1]
    pixels1 = 2*size1 + 1      ## Double and plus 1 to have an odd-sized kernel
    pixels2 = 2*size2 + 1      ## Double and plus 1 to have an odd-sized kernel
    kernel1 = np.ones((pixels1,pixels1),np.uint8)   ## Pixel of extension I get
    kernel2 = np.ones((pixels2,pixels2), np.uint8)                     ## Design an odd-sized erosion kernel
    dilation=cv2.dilate(image,kernel1,iterations=1)
    dilation=np.where(dilation == 255,127,dilation)
    erosion = cv2.erode(image, kernel2, iterations=1)  ## How many erosion do you expect
    erosion = np.where(erosion == 0, 127, erosion)
    remake = np.where(image == 0,0,erosion)
    remake = np.where(remake>0,remake,dilation)
    #path = "./trimap_{}/".format{size1}  ## Change the directory
    new_name = './trimap.png'#.format(size1)
    #print("success!")
    cv2.imwrite(new_name, remake)

def de(Image_file):
 # decision making:
 dec=[ 0,0,1,1,0,0,0,0,1,0,
      1,1,0,1,1,1,1,1,1,1,
      1,1,1,1,0,1,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,1,1,0,1,
      1,1,0,0,0,0,0,0,0,0,
      0,0,0,1,1,1,1,0,0,0,
      0,0,0,0,0,0,0,0,0,0]
 model = init_detector("configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py","checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth", device="cuda:1")

 # detect the object with given model and output the mask
 #Image_file='multi/7.jpeg'
 img=Image.open(Image_file)
 result = inference_detector(model, Image_file)
 bbox_results, mask_results=result
 tmp=np.zeros((img.size[1],img.size[0]))
 # 目前仅考虑对检测出来的第一个类别的mask进行颜色迁移
 Type = -1
 for ii in range(80):
    i = ii
    cur_mask=mask_results[i] 
    if (len(cur_mask)!=0)&(dec[i]!=0):
      #print(len(cur_mask))
      Type = i
      for j in range(len(cur_mask)):
      #j = 0
        for k in range(img.size[1]):
          for l in range(img.size[0]):
             if cur_mask[j][k,l]:
                tmp[k,l]=255 #以tmp记录mask
      break
 size1 = 15
 size2 = 15    #number = path[-5]
 unit01 = Toolbox(tmp);
 kernel1 = np.ones( (11,11), np.uint8 )
 opening = unit01.morph_close(tmp,kernel1)
 trimap(opening, size1,size2)
 #print(tmp)
 return Type
