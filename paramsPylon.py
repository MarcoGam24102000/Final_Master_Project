# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 13:39:25 2022

@author: Rui Pinto
"""

from pypylon import pylon
import numpy as np
import os

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
    
    while True:
    
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        converter = pylon.ImageFormatConverter()   
        
        camera.Open()
        
        print(str(camera.Open()))
        
        camera.Height.SetValue(image_height)
        camera.Width.SetValue(image_width)
        
        
        camera.CenterX=False 
        camera.CenterY=False   
        camera.AcquisitionFrameRateEnable.SetValue(True)
        camera.AcquisitionFrameRateAbs.SetValue(frame_rate)
        
        camera.GevSCPSPacketSize.SetValue(packet_size)
        
        # Inter-Packet Delay            
        camera.GevSCPD.SetValue(inter_packet_delay)
        
        # Bandwidth Reserve 
        camera.GevSCBWR.SetValue(bandwidth_reserv)
        
        # Bandwidth Reserve Accumulation
        camera.GevSCBWRA.SetValue(bandwidth_reserv_acc)  
        
        camera.StartGrabbing()
        camera.Open()
        
        camera.GainRaw = gain
        camera.ExposureTimeRaw = exp_time_us
        
        print("Configuration successful ... ") 