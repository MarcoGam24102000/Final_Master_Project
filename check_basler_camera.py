# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 10:38:07 2023

@author: marco
"""

def confirm_basler():

    from pypylon import pylon
    import keyboard
    import string
    import time
    
    alphabet = list(string.ascii_lowercase) + list(string.ascii_uppercase)
    
    basler = False
    
    tl_factory = pylon.TlFactory.GetInstance()
    
    model_name = ''
     
    again = True 
    
    cam = None
    for dev_info in tl_factory.EnumerateDevices():
        if dev_info.GetDeviceClass() == 'BaslerGigE':
            basler = True
            model_name = dev_info.GetModelName()
            print("using Basler camera %s @ %s" % (dev_info.GetModelName(), dev_info.GetIpAddress()))
            
   ##         while again == True:
            if True:
                
           #     try:
                    cam = pylon.InstantCamera(tl_factory.CreateDevice(dev_info))
                    basler = True
         #       except:
         #           print("Try again ...")
      ##              print("Waiting for key input ...")
                    
          #          basler = False
                    
                    # while True:
                    
                    #     for letter in alphabet:
                    #         if keyboard.is_pressed(letter):
                    #             again = True
                    #             hear = True
                    #             break 
                        
                    #     if hear:
                    #         break
                        
                    # if not again: 
                    #     time.sleep(3)
                    #     again = True
                        
                            
            break 
    else:
  ##      raise EnvironmentError("no GigE device found")
      print("No GigE device found")
    
    return basler, model_name

## basler = confirm_basler() 