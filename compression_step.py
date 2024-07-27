# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 19:23:25 2023

@author: marco
"""

import cv2
from PIL import Image
import numpy as np
import time

base_path = "C:\\Research_CiTechCare\\Work_15_12\\OtherImgs\\"

compression_x = []
times_comp_x =  []

timeImgs = []

quality = 50

for i in range(0,100):
    
    time1 = time.time()
    
    print("Analysing image " + str(i) + " ...")
    
    img = cv2.imread(base_path + "el_map_" + str(i) + ".png")   

    
    time2 = time.time()
    
    time_img = round(abs(time2-time1),5)
    
    timeImgs.append(time_img)
    
mean_time = np.mean(np.array([timeImgs]))
    
print("Mean time per image: " + str(mean_time) + " seconds")      
    

for i in range(0,100):
    
    print("Analysing image " + str(i) + " ...")
    
    img = cv2.imread(base_path + "el_map_" + str(i) + ".png")
    
    # cv2.imwrite("base_path" + "el_map_" + str(i) + ".jpeg", img)
    
    # img = cv2.imread(base_path + "el_map_" + str(i) + ".jpeg")
    
    time1 = time.time()
    
    image_compressed = Image.fromarray(img).save(base_path + "compressed_" + str(i) + ".jpg", "JPEG", quality=quality)
    
    time2 = time.time()    
   
    
#    print(image_compressed)
    
    compression_x.append(image_compressed)
    
    # cv2.imwrite(base_path + "compressed_img_" + str(i) + ".jpeg", image_compressed)
    
    
    time_comp_img = round(abs(time2-time1),5)
    times_comp_x.append(time_comp_img)
    
mean_comp_time = np.mean(np.array([times_comp_x]))
    
print("Mean compression time per image: " + str(mean_comp_time) + " seconds")


rate_time_red = round((mean_time/mean_comp_time)*100,5)

print("Mean Reduction rate for each image " + str(rate_time_red) + " %")
    
    