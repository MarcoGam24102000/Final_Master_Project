# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:39:12 2022

@author: Rui Pinto
"""

import PySimpleGUI as sg
import cv2
import time


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
    
  
        
        
     
    
    
    
    
    
    
    
    