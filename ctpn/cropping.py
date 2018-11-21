from __future__ import print_function

import glob
import os
import shutil
import sys

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

sys.path.append(os.getcwd())
from lib.fast_rcnn.config import cfg, cfg_from_file
#from lib.networks.factory import get_network
#from lib.fast_rcnn.test import test_ctpn
from lib.fast_rcnn.test import _get_blobs
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from lib.rpn_msr.proposal_layer_tf import proposal_layer
from PIL import Image 

import rec_char.tools.classify_hangul as rchar
from scipy import ndimage


def crop_word(i, nm_cropped_image):
#def pred_from_img(image, train):
#        image = image

        #color_complete = cv2.imread("img/" + image + ".png")
        color_complete = cv2.imread(nm_cropped_image )
         
          
        height, width = color_complete.shape[:2]
        print("height - " +  str(height)+"width -"+ str(width)+'\n')

        max_height = 200
        if max_height < height:
              scaling_factor = max_height/float(height)
              resized = cv2.resize(color_complete,None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        elif  height >10 :
              print("resize color complete: up")
              scaling_factor = max_height/float(height)
              resized = cv2.resize(color_complete,None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)
              color_complete = resized


#       print(("read", "img/" + image + ".png"))
        # read the bw image
#       gray_complete = cv2.imread("img/" + image + ".png", 0)
        gray_complete = cv2.imread(nm_cropped_image, 0)

        height, width = gray_complete.shape[:2]
        print("height - " +  str(height)+"width -"+ str(width)+'\n')

        max_height = 200
        if max_height < height:
              scaling_factor = max_height/float(height)
              resized = cv2.resize(gray_complete, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
              gray_complete = resized
        elif max_height >height and height >10 :
              print("resize gray complete: up")
              scaling_factor = max_height/float(height)
              resized = cv2.resize(gray_complete,None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)
              gray_complete = resized



        # better black and white version
        _, gray_complete = cv2.threshold(255-gray_complete, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        print (nm_cropped_image+'\n') 

        if not os.path.exists(nm_cropped_image):
                 os.makedirs(nm_cropped_image+'_/')
        os.makedirs(nm_cropped_image+str(i)+'_/')

#       cv2.imwrite("pro-img/compl.png", gray_complete)

        digit_image = -np.ones(gray_complete.shape)

        height, width = gray_complete.shape
        print("height - " +  str(height)+"width -"+ str(width)+'\n')

        """
        crop into several images
        """
        for cropped_width in range(150, 250, 30):
                for cropped_height in range(150, 350, 30):
                        for shift_x in range(0, width-cropped_width, int(cropped_width/4)):
                                for shift_y in range(0, height-cropped_height, int(cropped_height/4)):
                                        gray = gray_complete[shift_y:shift_y+cropped_height,shift_x:shift_x + cropped_width]
                                        if np.count_nonzero(gray) <= 55:
                                                 continue

                                        if (np.sum(gray[0]) != 0) or (np.sum(gray[:,0]) != 0) or (np.sum(gray[-1]) != 0) or (np.sum(gray[:,

                                                                              -1]) != 0):
                                                continue

                                        top_left = np.array([shift_y, shift_x])
                                        bottom_right = np.array([shift_y+cropped_height, shift_x + cropped_width])

                                        while np.sum(gray[0]) == 0:
                                                top_left[0] += 1
                                                gray = gray[1:]

                                        while np.sum(gray[:,0]) == 0:
                                                top_left[1] += 1
                                                gray = np.delete(gray,0,1)

                                        while np.sum(gray[-1]) == 0:
                                                bottom_right[0] -= 1
                                                gray = gray[:-1]


                                        while np.sum(gray[:,-1]) == 0:
                                                bottom_right[1] -= 1
                                                gray = np.delete(gray,-1,1)

                                        actual_w_h = bottom_right-top_left
                                        if (np.count_nonzero(digit_image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]+1) >
                                                                0.2*actual_w_h[0]*actual_w_h[1]):
                                                continue


                                        rows,cols = gray.shape
                                        compl_dif = abs(rows-cols)
                                        half_Sm = int(compl_dif/2)
                                        half_Big = half_Sm if half_Sm*2 == compl_dif else half_Sm+1
                                        if rows > cols:
                                                gray = np.lib.pad(gray,((0,0),(half_Sm,half_Big)),'constant')
                                        else:
                                                gray = np.lib.pad(gray,((half_Sm,half_Big),(0,0)),'constant')

                                        gray = cv2.resize(gray, (56, 56))
                                        gray = np.lib.pad(gray,((4,4),(4,4)),'constant')


                                        shiftx,shifty = getBestShift(gray)
                                        shifted = shift(gray,shiftx,shifty)
                                        gray = shifted

                                        #cv2.imwrite("pro-img/"+image+"_"+str(shift_x)+"_"+str(shift_y)+".png", gray)
                                        cv2.imwrite(nm_cropped_image+str(i)+"_/"+str(shift_x)+"_"+str(shift_y)+".png", gray)
                                        print("saved nm_cropped_image"+nm_cropped_image+str(i)+"_/"+"_"+str(shift_x)+"_"+str(shift_y)+".png")

                                        #classify(
                                        rchar.classify (nm_cropped_image+str(i)+"_/"+str(shift_x)+"_"+str(shift_y)+".png")



def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)
    print(cy,cx)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty



def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted



def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f





