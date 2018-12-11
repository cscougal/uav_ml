# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 16:56:35 2018

@author: CS08
"""


import sys
sys.path.append(r"C:\Users\cs08\Documents\Projects\Python_Scripts\RPA")
import ml_functions as uv



                       
ml_model = r"E:\RPA_ANALYSIS\COMBWICH\20170314\F1\ML\lenet\model\marker.model"
input_image = r"E:\RPA_ANALYSIS\COMBWICH\20170314\F1\RawImages\DSC00341.JPG"


#thresholds for detecting yellow markers in blue , green , red

bgr_low = [130, 200, 200]
bgr_high = [255, 255, 255]

blur_radius = 19

contours = uv.detectMarkers(input_image,blur_radius,bgr_low,bgr_high)
markers_x,markers_y = uv.classifyMarkers(input_image,contours,ml_model)
