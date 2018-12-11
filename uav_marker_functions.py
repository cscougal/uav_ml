# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 10:41:08 2018

@author: CS08
"""

import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras import backend as k

#lenet ml code adapted from Adrian Rosebrock @ PyImageSearch



def detectMarkers(input_image,blur_radius,lower_bgr_thresh,
                  upper_bgr_thresh,area_filter=200):
    
    '''This function thresholds an image according to upper and lower
        boundaries. All pixels outwith range will be masked and contours will
        be created for each valid object. These can then be filtered based 
        on area. Note colour space is defined as BGR.'''


    image = cv2.imread(input_image)
    show_im = image.copy()
    
    image_blur = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
    #Applies guassian Blur = easier detection of yellow/custom colour areas
    
    lower = np.array(lower_bgr_thresh, dtype = "uint8")
    upper = np.array(upper_bgr_thresh, dtype = "uint8")
    
    mask = cv2.inRange(image_blur, lower, upper)
    #filters to values of interest
    #output = cv2.bitwise_and(image_blur, image_blur, mask = mask)
       
    my_image ,contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)
    #only intereseted in contours
    
    sub_contours = [i for i in contours if cv2.contourArea(i) <= area_filter]
    #filters by area
    
    cv2.drawContours(show_im, sub_contours, -1, (36,28,237), 5)
    #specifiy colours etc of detected contours
    
    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1280,960)
    #dealing with large images so resze for visualisation
    
    cv2.imshow("image",show_im)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    return sub_contours


  

def classifyMarkers(input_image,contours,input_model,crop_window=14):
    
    '''This function takes contours(potential marker areas) applies a 
        crop window (28x28 by default) same as ML training architecture then 
        applies the lenet ML algorithm to determine proability of image being 
        a marker. Pixel coordinates of the marker are then returned'''
          
    marker_x, marker_y = [] , []

    for cont in contours:
        
        try:
    
            m=cv2.moments(cont) 
            
            cx=int(m['m10']/m['m00'])
            cy=int(m['m01']/m['m00']) 
            
            #givs us the centroid of each contour
                
            start_row = cx - crop_window
            end_row = cx  +  crop_window
             
            start_col = cy - crop_window
            end_col = cy  + crop_window   
             
            im = cv2.imread(input_image)
            im_crop= im[start_col:end_col,start_row:end_row]
            #crop image to window bounds via numpy slicing
                        
            im = im_crop.astype("float") / 255.0
            #Normalize image
            im = img_to_array(im)
            im = np.expand_dims(im, axis=0)
            #reshape input image to format tensorflow allows\trained upon
            
            model = load_model(input_model)
            #load keras/tensorflow model
             
            (other, marker) = model.predict(im)[0]
            # predicts probability whether image is of a marker or not
                       
            prob = marker if marker > other else other
            #determines which proability is >
                       
            if prob == marker:
                
                marker_x.append(cx)
                marker_y.append(cy)
                #if marker detected, retrun centroid of marker from image
                
            else:
                pass
            
            k.clear_session()
            #vital comamnd for ensuring optimum speed. 
            #Else each teration takes significantly longer
            
        except ZeroDivisionError:
            pass     
        
    return marker_x, marker_y    

  
  
