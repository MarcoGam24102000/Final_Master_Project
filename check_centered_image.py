# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 22:53:40 2023

@author: marco
"""

import numpy as np
import cv2

ok_list = []

ok_var = [-1,-1,-1,-1]

# img = cv2.imread("C:\\Research_CiTechCare\\video_image242.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

base_dir = 'C:\\Users\\Other\\files_python\\py_scripts\\ffmpeg-5.0.1-full_build\\bin\\GUI\\'
    

def black_thresh_square_defs():
    import PySimpleGUI as sg
    
    
    def_black_thresh = 50
    
    layout = [
        
        [sg.Text('Black threshold for black squares definition: ')],
        [sg.Slider(default_value = def_black_thresh, orientation ='horizontal', key='-BLACK_THRESH_SQUARE-', range=(0,100)),
         sg.Text(size=(5,2), key='')],
        [sg.Button('Next')]        
    ]
    
    window = sg.Window('Black Threshold for camera centering helping routine', layout) 
    
 ##   while True:
     
    if True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
   ##         break
            print("Exiting ...")
        if event == "Next":
            def_black_thresh = int(values['-BLACK_THRESH_SQUARE-']) 
   ##         break  
            print("Reading new value ...")
        
    
    window.close()
    
    return def_black_thresh  
    
    

def black_points_lim_gui():
    
    import PySimpleGUI as sg
    
    def_perc_black_dots = 60
    
    layout = [
        
        [sg.Text('Black points per square (%):')],
        [sg.Slider(default_value = def_perc_black_dots, orientation ='horizontal', key='-BLACK_DOTS-', range=(0,100)),
         sg.Text(size=(5,2), key='')],
        [sg.Button('Next')]        
    ]
    
    
    window = sg.Window('Black Points Per Square', layout)
    
    if True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
       ##     break
           print("Exiting ...")
        if event == "Next":
             def_perc_black_dots = int(values['-BLACK_DOTS-']) 
      ##       break 
             print("Reading new value ...")
    
    window.close()  
    
    return def_perc_black_dots

def set_ok(ok):
     ok_var = ok 
 ##    ok_list.append(ok_var)
 
     ok_new = []
 
     for okv in ok_var:
         ok_new.append(str(okv))
    
     ok_var = ok_new 
 
     print("Set ...")
 ## base_dir + 
     with open(base_dir + 'centering.csv','a', encoding='utf-8') as file:
            new_str = ', '.join(ok_var)
            file.write(new_str + '\n')
        
            print("Another line written to csv file -- estimating the best camera position")    
     
def get_ok():
     
     print("Fetching file with centering info ...")
    ## base_dir + 
     file = open(base_dir + 'centering.csv', 'r')
     
     info_pos_cam = []
     
     for file_line in file:
         str_f = str(file_line.read())
         
         str_f_splitted = str_f.split(', ')
         
         info_pos_cam.append(str_f_splitted)  
         
     
     
     return info_pos_cam
 

def help_camera_center(img, black_thresh, black_points_perc):  

    x_sq_dim = int(len(img)/3)
    y_sq_dim = int(len(img[0])/3)
    
    ok = [1, 1, 1, 1]           
    
        
    black_points_perc_decim = round(float(black_points_perc/100), 3)    
     
    fit_one = False 
    fit_two = False
    fit_three = False
    fit_four = False
    
    sq_side_img = np.zeros((x_sq_dim, y_sq_dim))
    
    img_left_top = img[0:int(len(img)/3), 0:int(len(img[0])/3)] 
    img_right_top = img[0:int(len(img)/3), int(len(img[0])*(2/3)):] 
    img_left_bottom = img[int(len(img)*(2/3)):, 0:int(len(img[0])/3)]
    img_right_bottom = img[int(len(img)*(2/3)):, int(len(img[0])*(2/3)):]
    
    if (img_left_bottom.shape)[0] > (img_left_top.shape)[0]:
        img_left_bottom = img_left_bottom[1:,:]
        
    if (img_right_bottom.shape)[0] > (img_right_top.shape)[0]:
        img_right_bottom = img_right_bottom[1:,:]
    
    number_black_points = 0
    
    for b in range(len(sq_side_img[0])):
        for a in range(len(sq_side_img)):
            if img_left_top[a,b] < black_thresh:     ## 50
                number_black_points += 1
                
    if number_black_points > black_points_perc_decim*(len(sq_side_img)*len(sq_side_img[0])):
        fit_one = True
                
        
    number_black_points = 0        
            
    for b in range(len(sq_side_img[0])):
        for a in range(len(sq_side_img)):
            if img_right_top[a,b] < black_thresh:   ## 50
                number_black_points += 1
    
    if number_black_points > black_points_perc_decim*(len(sq_side_img)*len(sq_side_img[0])): 
        fit_two = True
                
                
    number_black_points = 0      
            
    
    for b in range(len(sq_side_img[0])):
        for a in range(len(sq_side_img)):
            if img_left_bottom[a,b] < black_thresh:
                number_black_points += 1
    
    if number_black_points > black_points_perc_decim*(len(sq_side_img)*len(sq_side_img[0])): 
        fit_three = True
            
             
    number_black_points = 0        
    
    for b in range(len(sq_side_img[0])):
        for a in range(len(sq_side_img)): 
            if img_right_bottom[a,b] < black_thresh:
                number_black_points += 1 
    
    if number_black_points > black_points_perc_decim*(len(sq_side_img)*len(sq_side_img[0])): 
        fit_four = True
        
    
    if fit_one and fit_two and fit_three and fit_four:
        print("Image centered !!!")
        ok = [1,1,1,1]
    else:
        print("Image not centered !!!")
        ok = False
        
        if fit_one:
            print("Square box from left top right")            
        else:
            print("Square box from left top not found")
            ok[0] = 0
            
        if fit_two:
            print("Square box from right top right")
        else:
            print("Square box from right top not found")
            ok[1] = 0
        
        if fit_three:
            print("Square box from left bottom right")
        else:
            print("Square box from left bottom not found")
            ok[2] = 0
            
        if fit_four:
            print("Square box from right bottom right")
        else:
            print("Square box from right bottom not found")  
            ok[3] = 0
            
 #   set_ok(ok)
        
    return ok 

            
        
        





