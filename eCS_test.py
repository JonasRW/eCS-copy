'''
Created on 24.07.2018

@author: Stromer, Daniel
'''

import cv2
import numpy as np
from skimage import img_as_ubyte,io
from skimage.transform import rotate
from fftFunction import fft
from newROI import createROI
from adaptedFrangi import frangi 
import time

def perform_eCS_algorithm(img_in, sigma_x = 1.0, sigma_y = 1.0, angle = 0.0, mask_fft = np.zeros((0))):
    """
    Perform eCS algorithm described in:
    Enhanced Crack Segmentation (eCS): A Reference Algorithm for Segmenting 
    Cracks in Multicrystalline Silicon Solar Cells

    Parameters
    ----------
    img_in : ndarray
        Input image.
    angle : float
        Rotation angle in degrees in counter-clockwise direction.
    sigma_x : float
        Gaussian filter sigma in x-direction
    sigma_y: float
        Gaussian filter sigma in y-direction
    mask_fft : ndarray
        mask for fft filtering to get rid of grid fingers

    Returns
    -------
    img_result : ndarray
        processed image

    """

    if(mask_fft.size != 0):
        img_in = fft(img_in, mask_fft) 

    img_equalized = cv2.equalizeHist(img_in)
    img_roi = createROI(img_equalized)
    plt.figure()
    #plt.imshow(img_roi) - Good estimation of ROI. Implement in dataArtist
    img_smoothed= cv2.bilateralFilter(img_equalized,3,25,25)
    
    if angle != 0.0:
        img_smoothed = rotate(img_smoothed,angle,resize = True, preserve_range=True)
    
    img_result = img_as_ubyte(frangi(img_smoothed,sigma_x = sigma_x, sigma_y = sigma_y, beta1=0.5, beta2=0.05,black_ridges=True))
    
    if(sigma_x != sigma_y):
        img_y  = img_as_ubyte(frangi(img_smoothed, sigma_x = sigma_y, sigma_y = sigma_x, beta1=0.5, beta2=0.05,black_ridges=True))
        img_result = np.maximum(img_result,img_y)

    if angle != 0.0: #Modified to handel rect img of diff sizes with np.shape    
        w = img_result.shape[0]
        w_org = img_in.shape[0] #changed to higt and width
        offset = (w - w_org)//2
        h_org = img_in.shape[1]
        
        #img is rotated back, and scaled to right size
        img_result = rotate(img_result,-angle,resize=True,preserve_range=True)[offset:offset+w_org,offset:offset+h_org] 
        
    img_result = img_result*img_roi

    return img_result.astype('float32')#does this need to be float32? can unit8?


'''
Main method to run the program
'''
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from skimage import img_as_float
    from skimage.color import rgb2gray
    #Image path: change to load own images
    img_path = 'RawImages/maske+crop-savetest.png' #crack_image.tif
    
    #FFT path: path to fft mask
    fft_mask_path = 'Filters/fft_mask.tif'
    
    #eCS parameter - if degree_of_asymmetry is 1.0, it is basically standard Frangi filtering with an isotropic kernel 
    sigma_a = 2.0
    degree_of_asymmetry = 1.5

    #read files from paths
    mask_fft = io.imread(fft_mask_path)
    img_raw = io.imread(img_path)
    m = np.amax(img_raw)
    #Make sure they are gray scale and singel chanel so code can handle them.
    mask_fft = rgb2gray(img_as_float(mask_fft))
    img_raw = rgb2gray(img_as_float(img_raw))


    # perform algorithm
    result = perform_eCS_algorithm(img_raw,sigma_x = sigma_a, sigma_y = degree_of_asymmetry*sigma_a, angle= 0.0, mask_fft=mask_fft)
    plt.figure()
    plt.imshow(result)
    plt.title('Sigma:{}'.format(sigma_a) + ' and DoAs:{}'.format(degree_of_asymmetry))
    plt.savefig('plt_ecs_result.tif')
    
    #Image to unit8 for image storage
    #Does not give optimal image. might be dataloss when converted to 8bit.
    #uint8 from fftFunction
    result = cv2.normalize(result, 0, 255, cv2.NORM_MINMAX)
    result *=255
    result = np.uint8(result)
    #result = (result * m).round().astype(np.uint8)
    io.imsave('ecs_result.tif', result)
    cv2.imwrite('cv2_ecs_result.tif',result)

