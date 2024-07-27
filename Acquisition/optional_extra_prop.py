# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 08:57:40 2023

@author: Rui Pinto
"""

def optional_prop():
    import PySimpleGUI as sg 
    
    enab = []
    
    layout = [
        [sg.Text("Enable PFS validation routine \t\t"), sg.T("         "), sg.Checkbox('', default=False, key="-IN1-")],
        [sg.Text("Enable image acquisition timestamps saving \t"), sg.T("         "), sg.Checkbox('', default=False, key="-IN2-")],
        [sg.Button("Next")]  
    ]           
     
    window = sg.Window('Aditional options', layout)              
      
    while True:

       event, values = window.read()
      
       
       if event == "Exit" or event == sg.WIN_CLOSED:
           break
       
       if event == "Next":
           enable_val_pfs = values['-IN1-']
           enable_save_timestamps = values['-IN2-']
           
           enab = [enable_val_pfs, enable_save_timestamps]
           
           break
      
    window.close()
    
    return enab

def optional_prop_norm():
    import PySimpleGUI as sg 
    
    enable_save_timestamps = False
    
    layout = [       
        [sg.Text("Enable image acquisition timestamps saving \t"), sg.T("         "), sg.Checkbox('', default=False, key="-IN1-")],
        [sg.Button("Next")]  
    ]           
     
    window = sg.Window('Aditional options', layout)              
      
    while True:

       event, values = window.read()
      
       
       if event == "Exit" or event == sg.WIN_CLOSED:
           break
       
       if event == "Next":
           enable_save_timestamps = values['-IN1-']         
           
           break 
      
    window.close()
    
    return enable_save_timestamps
 
## enab = optional_prop()
    