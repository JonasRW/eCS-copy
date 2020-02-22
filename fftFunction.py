'''
Created on 07.02.2018

@author: oezkan
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

def fft(img,mask):
    """
    Perform fft filtering

    Parameters
    ----------
    img : ndarray
        Input image.
    mask : mask
        fft image

    Returns
    -------
    normalizedImg : ndarray
        processed image

    """
    
    
    #convert the image to float32
    img = np.float32(img)
    shape = img.shape
    #convert the mask to float32 and scale to input image size
    mask = cv2.resize(np.float32(mask),(shape[1],shape[0]))
    
    #normalize image to 0..1
    mask = cv2.normalize(mask,0,1,cv2.NORM_MINMAX)
    # apply fft
    dft = cv2.dft(img,flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft) 
    
    fshift_real = dft_shift[:,:,0]*mask
    fshift_imaginary = dft_shift[:,:,1]*mask

    fshift=np.zeros((shape[0],shape[1],2))
    fshift[:,:,0] = fshift_imaginary
    fshift[:,:,1] = fshift_real 
    
    #apply inverse fft
    img_back = cv2.idft(fshift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    
    #uint8
    normalizedImg = cv2.normalize(img_back, 0, 255, cv2.NORM_MINMAX)
    normalizedImg *=255
    normalizedImg = np.uint8(normalizedImg)
    
    return normalizedImg