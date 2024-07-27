# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 23:16:05 2022

@author: marco
"""



def gui_software():

    import PySimpleGUIWeb as sg
    from pypylon import pylon
    import numpy as np
    import os
    
    itemHeight = ["135", "240", "270", "480", "540", "960", "1080", "1920"]   
    repeat = True
    
    def setParametersToPypylon(values):
        print("Setting parameters to Pylon ...")
        gain = values['GAIN']
        black_level = values['BLACK_THRESH']    
        image_height = values['FIRST']                          
        image_width = values['SEC']
        exp_time_us = values['EXP_TIME']    
        packet_size = values['PACKET_SIZE']    
        inter_packet_delay = values['INTER_PACKET_DELAY']    
        frame_rate = values['FRAME_RATE']    
        bandwidth_reserv_acc = values['BANDWIDTH_RESV_ACC']
        bandwidth_reserv = values['BANDWIDTH_RESV']  
        
        print("gain: " + str(gain))
        print("black_level: " + str(black_level))
        print("image_height: " + str(image_height))
        print("image_width: " + str(image_width))   
        print("exp_time_us: " + str(exp_time_us))
        print("packet_size: " + str(packet_size))
        print("inter_packet_delay: " + str(inter_packet_delay))
        print("frame_rate: " + str(frame_rate))
        print("bandwidth_reserv_acc: " + str(bandwidth_reserv_acc))
        print("bandwidth_reserv: " + str(bandwidth_reserv))     
        
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        converter = pylon.ImageFormatConverter()   
        
        camera.Open()
        
        print(str(camera.Open()))
        
        camera.Height.SetValue(image_height)
        camera.Width.SetValue(image_width)
        
        
        camera.CenterX=False 
        camera.CenterY=False
        
        camera.GainRaw = gain
        camera.ExposureTimeRaw = exp_time_us
        
        camera.AcquisitionFrameRateEnable.SetValue(True)
        camera.AcquisitionFrameRateAbs.SetValue(frame_rate)
        
        camera.GevSCPSPacketSize.SetValue(packet_size)
        
        # Inter-Packet Delay            
        camera.GevSCPD.SetValue(inter_packet_delay)
        
        # Bandwidth Reserve 
        camera.GevSCBWR.SetValue(bandwidth_reserv)
        
        # Bandwidth Reserve Accumulation
        camera.GevSCBWRA.SetValue(bandwidth_reserv_acc)  
        
        print("Configuration successful ... ")
          
    
    col = [
      [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
      ],
      [
        sg.Listbox(values=[], size=(50,50))
      ]
    ]
  
    col2 = sg.Column([         
         
            # Information frame
    [sg.Frame(layout=[[sg.Text('Gain:')],
                              [sg.Input(default_text= "10", size=(19, 1), key="GAIN")],
                              [sg.Text('Black Level:')],
                              [sg.Input(default_text= "2", size=(19, 1), key="BLACK_THRESH")],    ## , sg.Button('Copy')
                              [sg.Text('Image Height:')],
                              [sg.InputCombo(itemHeight, size=(19, 1), key="FIRST", change_submits=False) ],                         
                              [sg.Text('Image Width:')],
                              [sg.InputCombo(itemHeight, size=(19, 1), key="SEC", change_submits=False)],                          
                              [sg.Text('Exposition Time:')], 
                              [sg.Input(default_text= "140", size=(19, 1), key="EXP_TIME")]     ## , sg.Button('Copy')                            
                          
                              ], title='Information:')],])
        
    col3 = sg.Column([  
        
            # Information frame
            [sg.Frame(layout=[[sg.Text('Packet Size:')], 
                              [sg.Input(default_text= "1500", size=(19, 1), key="PACKET_SIZE")],     ## , sg.Button('Copy')
                              [sg.Text('Inter-Packet Delay:')],
                              [sg.Input(default_text= "5000", size=(19, 1), key="INTER_PACKET_DELAY")],
                              [sg.Text('Frame rate:')],
                              [sg.Input(default_text= "50", size=(19, 1), key="FRAME_RATE")],
                              [sg.Text('Bandwidth Reserve Accumulation:')],
                              [sg.Input(default_text= "4", size=(19, 1), key="BANDWIDTH_RESV_ACC")],
                              [sg.Text('Bandwidth Reserve:')],
                              [sg.Input(default_text= "10", size=(19, 1), key="BANDWIDTH_RESV")]                           
                              ], title='Information 2:')],])
            
    col1 =  sg.Column(col)
        
    layout = [ [col1, col2, col3],
            # Actions Frame 
            [sg.Frame(layout=[[sg.Button('Save'), sg.Button('Clear'), sg.Button('Delete'),                       
                                
                ]], title='Actions:')]]   
         
        # Position at top left side corner on right hand monitor 
    window = sg.Window('Passwords', layout, web_port=2255, web_start_browser=False, disable_close=True)   ## Always changing     
        
    valid = True 
        
    while repeat == True:             # Event Loop
            event, values = window.read()
            print(event, values)
            
            if event == 'Save':         
                if int(values["GAIN"]) < 0 or int(values["GAIN"]) > 50:
                    sg.popup_error('Gain must be within [0,50] interval') 
                    valid = False
                    print("Failed to gain range")
                if int(values["BLACK_THRESH"]) < 0 or int(values["BLACK_THRESH"]) > 255:
                    sg.popup_error('Black level must be within [0,255] interval')    ## for Mono8 configuration 
                    valid = False
                    print("Failed to black level range")
                if int(values["EXP_TIME"]) < 0 or int(values["EXP_TIME"]) > 35000:
                    sg.popup_error('Exposition time must be within [0,35000] interval')  
                    valid = False
                    print("Failed to exposure time range")
                if int(values["PACKET_SIZE"]) <= 0:
                    sg.popup_error('Packet size must be positive')
                    valid = False
                    print("Failed to packet size range")
                if int(values["INTER_PACKET_DELAY"]) < 0 or int(values["INTER_PACKET_DELAY"]) > 10000:
                    sg.popup_error('Inter-packet delay must be within [0,10000] interval') 
                    valid = False  
                    print("Failed to inter-packet delay range")                
                if int(values["FRAME_RATE"]) < 10 or int(values["FRAME_RATE"]) > 50: 
                    sg.popup_error('Frame rate (fps) must be within [10,50] interval') 
                    valid = False
                    print("Failed to frame rate range")
                if int(values["BANDWIDTH_RESV_ACC"]) < 0 or int(values["BANDWIDTH_RESV_ACC"]) > 10:
                    sg.popup_error('Bandwidth reserve accumulation must be within [0,100] interval')  
                    valid = False
                    print("Failed to bandwidth reserve accumalatio range")
                if int(values["BANDWIDTH_RESV"]) < 0 or int(values["BANDWIDTH_RESV"]) > 100:
                    sg.popup_error('Bandwidth reserve must be within [0,10] interval')   
                    valid = False
                    print("Failed to bandwidth reserve range")
                    
                if valid == True:
                    print("Next step")
                    values.pop(0)
                    repeat = False
              ##      setParametersToPypylon(values)   
              ## or just call the whole python file related to image processing and so on
                    break
                else:
                    repeat = True
                    valid = True
                    print("Try again")
                    del values, event
            else:
                if event == "-FOLDER-":
                    folder = values["-FOLDER-"]
                    try:                       
                        file_list = os.listdir(folder)
                    except:
                        file_list = []                   
             
    window.close() 
    
    return values

