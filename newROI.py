'''
Created on 16.03.2018

@author: oezkan
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

def mainBorder(img_in):
    '''
    This function extracts the main border.
    It has to be applied after histogram equalization!
    '''
    threshold_value = np.min(img_in)
    _,ROI = cv2.threshold(img_in,(threshold_value+65),255,cv2.THRESH_BINARY)
    invertROI = cv2.bitwise_not(ROI)
    
    #setting rectangle kernels
    rect_y = cv2.getStructuringElement(cv2.MORPH_RECT,(4,2000))
    rect_x = cv2.getStructuringElement(cv2.MORPH_RECT,(2000,4))
    
    rect_opened_y = cv2.morphologyEx(invertROI,cv2.MORPH_OPEN,rect_y)
    rect_opened_y = cv2.bitwise_not(rect_opened_y)
    
    rect_opened_x = cv2.morphologyEx(invertROI,cv2.MORPH_OPEN,rect_x)
    rect_opened_x = cv2.bitwise_not(rect_opened_x)
    
    rect = rect_opened_x * rect_opened_y
    
    kernel_x = np.ones((1,8),np.uint8)
    img_morph2 = cv2.morphologyEx(rect, cv2.MORPH_OPEN, kernel_x)
        
    kernel_y = np.ones((8,1),np.uint8)
    img_morph1 = cv2.morphologyEx(rect, cv2.MORPH_OPEN, kernel_y)
    
    img_morph = img_morph2 * img_morph1
    
    _,contours,_ = cv2.findContours(img_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    main_border = np.zeros((img_morph.shape))
    border = []
    for index in range(len(contours)):
        if(cv2.contourArea(contours[index]) > 100000):
            border.append(contours[index])
            cv2.drawContours(main_border, border, -1, 1, -1)
            
    epsilon = 0.1* cv2.arcLength(border[0],True)
    approx = cv2.approxPolyDP(border[0],epsilon,True)                    
    cv2.drawContours(main_border,approx,-1,1,-1)
    
    kernel_ROI = np.ones((8,8),np.uint8)
    main_border = cv2.erode(main_border,kernel_ROI,iterations = 1)
    
    return main_border

def busBars(img_in):
    '''
    This function extracts the main border and bus bars together.
    It has to be applied after histogram equalization!
    '''
    border = mainBorder(img_in)

    threshold_value = np.min(img_in)
    _,ROI = cv2.threshold(img_in,(threshold_value+70),255,cv2.THRESH_BINARY)
    invertROI = cv2.bitwise_not(ROI)

    invertROI *= border.astype(invertROI.dtype)
       
    rect_y = cv2.getStructuringElement(cv2.MORPH_RECT,(1,400))
    
    rect_opened_y = cv2.morphologyEx(invertROI,cv2.MORPH_OPEN,rect_y)
    rect_opened_y = cv2.bitwise_not(rect_opened_y)
    
    kernel_y = np.ones((4,16),np.uint8)
    rect_opened_y1 = cv2.erode(rect_opened_y,kernel_y,1)
    
    return rect_opened_y1


def createROI(img_in):
    '''
    This function gets main border and bus bars together.
    It needs to be applied on an image which has histogram equalization!!!!!!!!!!!!
    '''

    border = mainBorder(img_in)
    bus_bars = busBars(img_in)
    
    ROI = (bus_bars*border)/255
    
    return ROI
