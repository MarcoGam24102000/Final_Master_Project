# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 17:04:10 2023

@author: Rui Pinto
"""

import time

from pypylon_opencv_viewer import BaslerOpenCVViewer
from pypylon import pylon      
import cv2
import PySimpleGUI as sg


def further_reduction(image, init_dim):
    print("Further reduction")
    
    if len(image) % 2 == 0 and len(image[0]) % 2 == 0:
        small_dim = (len(image)/2, len(image[0])/2)
    else:
        if len(image) % 2 != 0 and len(image[0]) % 2 == 0:
            small_dim = ((len(image)-1)/2, len(image[0])/2)
        else:
            if len(image) % 2 == 0 and len(image[0]) % 2 != 0:
                small_dim = ((len(image))/2, (len(image[0])-1)/2)
            else:
                small_dim = (((len(image))-1)/2, (len(image[0])-1)/2) 
                
    small_dim = tuple(int(item) for item in small_dim)  
    
    resized_img = cv2.resize(image, small_dim, interpolation = cv2.INTER_AREA)    
    
    return small_dim, resized_img   
    

def showLiveImageGUI(image, numberImageSeqVideo):
    
    next = False
    
    print("Rendering " + str(numberImageSeqVideo) + " th image ...")
    
    cv2.imwrite("video_image" + str(numberImageSeqVideo) + ".png", image)
    
    if len(image) % 2 == 0 and len(image[0]) % 2 == 0:
        new_dim = (len(image)/2, len(image[0])/2)
    else:
        if len(image) % 2 != 0 and len(image[0]) % 2 == 0:
            new_dim = ((len(image)-1)/2, len(image[0])/2)
        else:
            if len(image) % 2 == 0 and len(image[0]) % 2 != 0:
                new_dim = ((len(image))/2, (len(image[0])-1)/2)
            else:
                new_dim = (((len(image))-1)/2, (len(image[0])-1)/2) 

    new_dim = tuple(int(item) for item in new_dim)               
                
    image = cv2.imread("video_image" + str(numberImageSeqVideo) + ".png")
    
    resized_img = cv2.resize(image, new_dim, interpolation = cv2.INTER_AREA)
    
    small_dim, resized_img = further_reduction(resized_img, new_dim)
    
    cv2.imwrite("micro_image" + str(numberImageSeqVideo) + ".png", resized_img)
    
    size_img = (len(resized_img), len(resized_img[0]))
    
    layout = [

        [sg.Image('micro_image' + str(numberImageSeqVideo) + '.png', size = size_img)],
        [sg.Button('Next'), sg.Button('Exit')]   ## 
    ]
     
    
    window = sg.Window('Images', layout, resizable = True, finalize = True, margins=(0,0)) 
    
    button,values = window.read(timeout=200)    ## 1000  
     
    window.close()
    

def show_live_image_camera(countx, trig):
    
    
    
    if countx == 0:
        
        print("BaslerOpenCVViewer configuration ...")
        
#         camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
# ##        converter = pylon.ImageFormatConverter()

#         camera.Open()  
        
#         viewer = BaslerOpenCVViewer(camera)
        
        # # Example of configuration for basic RGB camera's features
        # VIEWER_CONFIG_RGB_MATRIX = {
        #     "features": [
        #         {
        #             "name": "GainRaw",
        #             "type": "int",
        #             "step": 1,
        #         },
        #         {
        #             "name": "Height",
        #             "type": "int",
        #             "value": 1080,
        #             "unit": "px",
        #             "step": 2,
        #         },
        #         {
        #             "name": "Width",
        #             "type": "int",
        #             "value": 1920,
        #             "unit": "px",
        #             "step": 2,
        #         },
        #         {
        #             "name": "CenterX",
        #             "type": "bool",
        #         },
        #         {
        #             "name": "CenterY",
        #             "type": "bool",
        
        #         },
        #         {
        #             "name": "OffsetX",
        #             "type": "int",
        #             "dependency": {"CenterX": False},
        #             "unit": "px",
        #             "step": 2,
        #         },
        #         {
        #             "name": "OffsetY",
        #             "type": "int",
        #             "dependency": {"CenterY": False},
        #             "unit": "px",
        #             "step": 2,
        #         },
        #         {
        #             "name": "AcquisitionFrameRateAbs",
        #             "type": "int",
        #             "unit": "fps",
        #             "dependency": {"AcquisitionFrameRateEnable": True},
        #             "max": 150,
        #             "min": 1,
        #         },
        #         {
        #             "name": "AcquisitionFrameRateEnable",
        #             "type": "bool",
        #         },
        #         {
        #             "name": "ExposureAuto",
        #             "type": "choice_text",
        #             "options": ["Off", "Once", "Continuous"],
        #             "style": {"button_width": "90px"}
        #         },
        #         {
        #             "name": "ExposureTimeAbs",
        #             "type": "int",
        #             "dependency": {"ExposureAuto": "Off"},
        #             "unit": "μs",
        #             "step": 100,
        #             "max": 35000,
        #             "min": 500,
        #         },
        #         {
        #             "name": "BalanceWhiteAuto",
        #             "type": "choice_text",
        #             "options": ["Off", "Once", "Continuous"],
        #             "style": {"button_width": "90px"}
        #         },
        #     ],
        #     "features_layout": [
        #         ("Height", "Width"), 
        #         ("OffsetX", "CenterX"), 
        #         ("OffsetY", "CenterY"), 
        #         ("ExposureAuto", "ExposureTimeAbs"),
        #         ("AcquisitionFrameRateAbs", "AcquisitionFrameRateEnable"),
        #         ("BalanceWhiteAuto", "GainRaw")
        #     ],
        #     "actions_layout": [
        #         ("StatusLabel"),
        #         ("SaveConfig", "LoadConfig", "ContinuousShot", "SingleShot"), 
        #         ("UserSet")
        #     ],
        #     "default_user_set": "UserSet3",
        #     }
    
        # viewer.set_configuration(VIEWER_CONFIG_RGB_MATRIX)
        
        while True:
            
            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    ##        converter = pylon.ImageFormatConverter()

            camera.Open()  
            
            viewer = BaslerOpenCVViewer(camera)
        
            img = viewer.get_image() 
            
            print("Shape of camera image: " + str(img.shape))
     ##       time.sleep(0.01)
        #     cv2.imshow('Live image', img)
       
            showLiveImageGUI(img, 0)
        

        
        # try:
        
        #     img = viewer.get_image() 
        #     time.sleep(0.01)
        #     cv2.imshow('Live image', img)
        # except Exception:
        #     print("--- Skipping ...")
       
  ##      cv2.waitKey(0) 
        
    # if viewer:        

    #     if trig == True:
    #         print(" -- Showing camera image")
            
    #         img = viewer.get_image()        
    #         time.sleep(0.01)
    #         cv2.imshow('Live image', img)
    #         cv2.waitKey(0)
           
    #     else:
    #         print("Stopping live camera image. Going forward ...")
        
if True:
    show_live_image_camera(0, True) 
 
  