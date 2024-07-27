# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 22:42:01 2023

@author: marco
"""

from check_basler_camera import confirm_basler
import time

def show_test_image(img, test_img_filename):   
    
    import PySimpleGUI as sg
    import cv2
    
    img = cv2.imread(test_img_filename)
    
    size_img = (len(img), len(img[0])) 
    
    layout = [ 

        [sg.Image(test_img_filename, size = size_img)],
        [sg.Button('Skip')]   ## 
    ]
     
    
    window = sg.Window('Test image', layout, resizable = True, finalize = True, margins=(0,0)) 
    
    print("Got here ...")
    
    import time
    
    t1 = time.time()
    
    while True:
        event, values = window.read()
        t2 = time.time()
        
        if abs(t2-t1) > 5:
            break
    
    # while True:
    #     event, values = window.read()
        
    #     if event == "Exit" or event == sg.WIN_CLOSED:
    #           break
    #     if event == 'Skip': 
    #         break 
        
        
    window.close()     
    

def get_test_image(params_streaming):    
    
    from pypylon import pylon
    import os
    import cv2
    
    gain = int(params_streaming[0])
    black_lavel = int(params_streaming[1])
    format_image = params_streaming[2]
    expos_time = int(params_streaming[3])   
    
    packet_size = int(params_streaming[4])
    inter_packet_delay = int(params_streaming[5])
    bw_reserv_acc = int(params_streaming[6])
    bw_reserv = int(params_streaming[7])
    
    frame_rate = int(params_streaming[8]) 
    
    
    ####################################################################
    
    width = 1920
    height = 1080
    
    if format_image == 'Full HD':
        width = 1920
        height = 1080
    elif format_image == 'HD+':
        width = 1600
        height = 900
    elif format_image == 'HD':
        width = 1280
        height = 720    
    elif format_image == 'qHD':
        width = 960
        height = 540
    elif format_image == 'nHD':
        width = 640
        height = 360        
    elif format_image == '960H':
        width = 960
        height = 480 
    elif format_image == 'HVGA':
        width = 480
        height = 320          
    elif format_image == 'VGA':
        width = 640
        height = 480    
    elif format_image == 'SVGA':
        width = 800
        height = 600  
    elif format_image == 'DVGA':
        width = 960
        height = 640 
    elif format_image == 'QVGA':
        width = 320
        height = 240
    elif format_image == 'QQVGA':
        width = 160
        height = 120
    elif format_image == 'HQVGA':
        width = 240
        height = 160  
    
    #################################################################### 
    
    basler_check = False
    camera_opened = True
    
    while not basler_check:                                            
        basler_check, model_name = confirm_basler()
        time.sleep(2)    
      
        
    while camera_opened == True:
        try:
            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            converter = pylon.ImageFormatConverter()  
            camera.Open()
            camera_opened = False
        except:
            camera.Close()
            camera_opened = True 
    
    
    camera.CenterX=False
    camera.CenterY=False
           
          
           
    # Set the upper limit of the camera's frame rate to 30 fps
    camera.AcquisitionFrameRateEnable.SetValue(True)
    camera.AcquisitionFrameRateAbs.SetValue(frame_rate) 
           
    camera.GevSCPSPacketSize.SetValue(packet_size)
           
    # Inter-Packet Delay            
    camera.GevSCPD.SetValue(inter_packet_delay)
           
    # Bandwidth Reserve 
    camera.GevSCBWR.SetValue(bw_reserv)
           
    # Bandwidth Reserve Accumulation
    camera.GevSCBWRA.SetValue(bw_reserv_acc)    
           
    ## Save feature data to .pfs file
    ##  pylon.FeaturePersistence.Save(nodeFile, camera.GetNodeMap())            
       
    # demonstrate some feature access
    new_width = camera.Width.GetValue() - camera.Width.GetInc()
    if new_width >= camera.Width.GetMin():
          camera.Width.SetValue(new_width)
               
    camera.Width.SetValue(width) 
    camera.Height.SetValue(height)
    
    camera.StartGrabbing()           
    
    camera.Open()
           
    camera.GainRaw=gain
    camera.ExposureTimeRaw=expos_time
         
           
    while camera.IsGrabbing():            
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        
        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult)
            img = image.GetArray()  
                         ##          showLiveImageGUI(img, counter)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            break
        
    # cv2.imshow('Test camera image', img)
    # cv2.waitKey(1)  

    test_img_filename = "test_image.png"
    
    if os.path.isfile(test_img_filename):
        test_parts = test_img_filename.split('png')
        
        test_rem = ""        
        for d in test_parts:
            if len(d) != 0:
                test_rem = d
        
        ind_test = 0
        
        while os.path.isfile(test_img_filename):            
            ind_test += 1
        
            test_img_filename = test_rem + "_" + str(ind_test) + ".png"
            
        cv2.imwrite(test_img_filename, img)
            
    else:        
        cv2.imwrite(test_img_filename, img) 
    
    show_test_image(img, test_img_filename)
    
    
    
    
    
    

                            
    
    
    
    
    
