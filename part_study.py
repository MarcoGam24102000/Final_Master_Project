# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 14:37:42 2023

@author: Rui Pinto
"""

import cv2
from study_calib_params_aux import calib_camera_part
import string
import keyboard

full_dir = "C:\\Users\\Other\\files_python\\py_scripts\\ffmpeg-5.0.1-full_build\\bin\\GUI\\out_5_1_23_3\\image0_"

alphabet = list(string.ascii_lowercase) + list(string.ascii_uppercase)

counter_failed = 0
lim = 5

for i in range(1,251):
    
    print("Finding calibration parameters for chess image number " + str(i))
    
    img = cv2.imread(full_dir + str(i) + ".tiff")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # for b in range(len(img[0])):
    #     for a in range(len(img)):
    #         img[a,b] = 2*img[a,b]
    
    cv2.imwrite(full_dir + str(i) + ".png", img)   
    
    dir_img = full_dir + str(i) + ".png"
    
    key_calib = False
    print("Press a key ...")
    
    while key_calib == False:
    
        for letter in alphabet:
            if keyboard.is_pressed(letter):
                key_calib = True 
    
    if key_calib == True:
    
        code_ret = calib_camera_part(True, "Basler", dir_img)
        
        if code_ret == -2 and counter_failed == lim and i == lim+1:
            print("Calibration failed ...")
        elif code_ret == -2:
            counter_failed += 1
    
    
    