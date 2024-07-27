# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 15:18:25 2022

@author: marco
"""

import cv2
import glob
import os


#####################################################################

def sel_lim_info():
    
    import PySimpleGUI as sg
    number_contours = 0
    
    ## Select Lim Info
    
    col = sg.Column([       
             
                # Information frame
        [sg.Frame(layout=[[sg.Text('Threshold number of contours:')], 
                                  [sg.Input(default_text= "5000", size=(19, 1), key="Number_C")],
                                 
                                  ], title='Information for classification of first image:')
        ],])   
    
    layout = [ [col],     
                # Actions Frame 
                [sg.Frame(layout=[[sg.Button('Next')]], title='Actions:')]]
    
    window = sg.Window('GUI', layout, disable_close=True, resizable = True, finalize = True, margins=(0,0))  
    
    valid = True  
    
    while True:
        
        event, values = window.read()
        print(event, values)        
              
                 
        if event == "Exit" or event == sg.WIN_CLOSED:
              break
        elif event == "Next":
            if len(values["Number_C"]) == 0:
                  sg.popup_error('Threshold number of contours - Empty field') 
                  valid = False
            
            if valid == True:  
                number_contours = int(values["Number_C"])
                break
    
    window.close()
    
    return number_contours    

def type_img(filename, base_path, idImg):
    
    code_img = 0
    
    lim_info = sel_lim_info()
    
##    lim_info = 5000
    
    img = cv2.imread(filename)
    
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_rev = cv2.bitwise_not(img_grey)
    
    cv2.imwrite(base_path + '/test_img_' + str(idImg) + '.tiff', img_rev)  
    
    img_rev =  cv2.imread(base_path + '/test_img_' + str(idImg) + '.tiff')        
    
    ret, binary = cv2.threshold(img_rev,40,255,cv2.THRESH_BINARY)
    binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)   
    
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    print("Number of countours: " + str(len(contours)))
    
    if len(contours) < lim_info:
        code_img = 1               
    else:
        code_img = 2
    
    return code_img 



def classifiy_image(filename):

    def sel_lim_info():
        
        import PySimpleGUI as sg
        number_contours = 0
        
        ## Select Lim Info
        
        col = sg.Column([       
                 
                    # Information frame
            [sg.Frame(layout=[[sg.Text('Threshold number of contours:')], 
                                      [sg.Input(default_text= "5000", size=(19, 1), key="Number_C")],
                                     
                                      ], title='Information for classification of first image:')
            ],])   
        
        layout = [ [col],     
                    # Actions Frame 
                    [sg.Frame(layout=[[sg.Button('Next')]], title='Actions:')]]
        
        window = sg.Window('GUI', layout, disable_close=True, resizable = True, finalize = True, margins=(0,0))  
        
        valid = True  
        
        while True:
            
            event, values = window.read()
            print(event, values)        
                  
                     
            if event == "Exit" or event == sg.WIN_CLOSED:
                  break
            elif event == "Next":
                if len(values["Number_C"]) == 0:
                      sg.popup_error('Threshold number of contours - Empty field') 
                      valid = False
                
                if valid == True:  
                    number_contours = int(values["Number_C"])
                    break
        
        window.close()
        
        return number_contours    
        
    
    def type_img(filename, base_path, idImg):
        
        code_img = 0
        
        lim_info = sel_lim_info()
        
    ##    lim_info = 5000
        
        img = cv2.imread(filename)
        
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img_rev = cv2.bitwise_not(img_grey)
        
        cv2.imwrite(base_path + '/test_img_' + str(idImg) + '.tiff', img_rev)  
        
        img_rev =  cv2.imread(base_path + '/test_img_' + str(idImg) + '.tiff')        
        
        ret, binary = cv2.threshold(img_rev,40,255,cv2.THRESH_BINARY)
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)   
        
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < lim_info:
            code_img = 1               
        else:
            code_img = 2
        
        return code_img 


#####################################################################

# base_path = 'C:/Research/PreClassifying/Core'

# os.chdir(base_path)

    counterChessImg = 0
    counterLaserSpeckleImg = 0
    
    numberROIS_chess = []
    numberROIS_laser_speckle = []

## for filename in glob.glob(base_path + '/*tiff'):

    dirx_splitted = filename.split('/')
    dirx_splitted = dirx_splitted[:-1]
    base_path = ''

    for d in dirx_splitted:
        base_path += d + '/'
    
    code_img = type_img(filename, base_path, 1)
    
    if code_img == 0: 
        print("Error classifying image !!!")
    elif code_img == 1:
        print("Laser Speckle Image detected !!!")
    elif code_img == 2:
        print("Chess Image detected !!!")
    
    
    # if 'image0_0_' in filename:
        
    #     print("Analysing chess image number " + str(counterChessImg))
    #     counterChessImg += 1
        
    #     chess_img = cv2.imread(filename)
        
    #     chess_grey = cv2.cvtColor(chess_img, cv2.COLOR_BGR2GRAY)
        
    #     chess_rev = cv2.bitwise_not(chess_grey)
        
    #     cv2.imwrite(base_path + '/rev_chess_img_' + str(counterChessImg-1) + '.tiff', chess_rev)  
        
    #     chess_rev =  cv2.imread(base_path + '/rev_chess_img_' + str(counterChessImg-1) + '.tiff')        
        
    #     ret, binary = cv2.threshold(chess_rev,40,255,cv2.THRESH_BINARY)
    #     binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)        
        
        
    #     contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
    #     print("Dims for rois within chess image: ")        
        
    #     number_rois = len(contours)  
    #     numberROIS_chess.append(number_rois)
        
    #     for ind_c, c in enumerate(contours):
    #         x,y,w,h = cv2.boundingRect(c)
            
    #         print(str(ind_c) + "ª ROI ... ")
            
            
    #         print("(" + str(x) + " , " + str(y) + " , " + str(w) + " , " + str(h) + ")")
            
        
        
    # elif 'image0_1_' in filename:
        
    #     print("Analysing  number " + str(counterLaserSpeckleImg))        
    #     counterLaserSpeckleImg += 1
        
    #     laser_speckle_img = cv2.imread(filename)
        
    #     laser_speckle_grey = cv2.cvtColor(laser_speckle_img, cv2.COLOR_BGR2GRAY)

    #     laser_speckle_rev = cv2.bitwise_not(laser_speckle_grey)
        
    #     cv2.imwrite(base_path + '/rev_laser_speckle_img_' + str(counterLaserSpeckleImg-1) + '.tiff', laser_speckle_rev) 
        
    #     laser_speckle_rev =  cv2.imread(base_path + '/rev_laser_speckle_img_' + str(counterChessImg-1) + '.tiff')        
        
    #     ret, binary = cv2.threshold(laser_speckle_rev,40,255,cv2.THRESH_BINARY) 
    #     binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)        
        
    #     contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
    #     print("Dims for rois within laser speckle image: ")
        
    #     number_rois = len(contours)
    #     numberROIS_laser_speckle.append(number_rois)        
        
    #     for ind_c, c in enumerate(contours):
    #         x,y,w,h = cv2.boundingRect(c) 
            
    #         print(str(ind_c) + "ª ROI ... ")
            
            
    #         print("(" + str(x) + " , " + str(y) + " , " + str(w) + " , " + str(h) + ")")