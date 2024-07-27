# -*- coding: utf-8 -*-
"""
Created on Sat May 13 15:44:33 2023

@author: marco
"""

from check_basler_camera import confirm_basler
from open_form_gui import exp_control  
from optional_extra_prop import optional_prop, optional_prop_norm 
from pfs_input import read_pfs_file 
import time

startTime = time.time() 
 
adv_written = False 

itemHeight = ["135", "240", "270", "480", "540", "960", "1080", "1920"]


def repeat_loop_proc(): 
    
    import PySimpleGUI as sg
    
    layout = [ 
        [sg.Text("One more ?")],
        [sg.T("         "), sg.Checkbox('Yes', default=False, key="-IN1-")],
        [sg.T("         "), sg.Checkbox('No', default=True, key="-IN2-")],
        [sg.Button("Exit"), sg.Button("Next")]
    ] 
    
    window = sg.Window("Repeat Loop", layout)
    
    again = True 
    proc_again = 0
    
    while again == True:
        event, values = window.read()
        
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == "Next":
            if values["-IN1-"] and values["-IN2-"]:
                print("Please select only one option ...")
                again = True
            elif not values["-IN1-"] and not values["-IN2-"]:
                print("Select one option ...")
                again = True
            else:
                if values["-IN1-"] and not values["-IN2-"]:
                    print("One more ...")
                    proc_again = 2
                elif values["-IN2-"] and not values["-IN1-"]:
                    print("Ending ...")
                    proc_again = 1
                    
                break
            
    window.close() 
    
    return proc_again   

resTypes = ['Full HD',
            'HD+',
            'HD',
            'qHD',
            'nHD',
            '960H',
            'HVGA',
            'VGA',
            'SVGA',
            'DVGA',
            'QVGA',
            'QQVGA',
            'HQVGA'
           ]

res_dims = ['(1920x1080)',
            '(1600x900)',
            '(1280x720)',
            '(960x540)',
            '(640x360)',
            '(960x480)',
            '(480x320)',
            '(640x480)',
            '(800x600)',
            '(960x640)',
            '(320x240)',
            '(160x120)',
            '(240x160)'
          ]   

res_fullTypes = []

for ind_r, r in enumerate(resTypes):
    r += ' ' + res_dims[ind_r]
    res_fullTypes.append(r)


basler_check = False

basler_check, model_name = confirm_basler() 

if basler_check == True:
    
    from pfs_input_acq_step_only import read_pfs_file
    
    repeat = True  
    
    while True:
        control_inf = exp_control()
        
        if control_inf[0] == True:
            
            numberTests = control_inf[1]
            gap_bet_tests = control_inf[2]
            
            break
        else:
            continue
    
    print("Number of tests: " + str(numberTests))
        
    if numberTests > 1: 
        print("Going to perform for " + str(numberTests) + " tests with a " + str(gap_bet_tests) + " minutes gap")
    else:
        print("Going to perform just 1 test")
    
    print("numTests init: " + str(numberTests))
    
    enab = optional_prop()
     
    enab_val_pfs = enab[0]
    enab_timestamps = enab[1]
    
    print(" \n Gap: " + str(read_pfs_file) + "\n")
    
    read_pfs_file(gap_bet_tests, numberTests, repeat, startTime, enab_val_pfs, enab_timestamps)
 
resp_again = 2 

while resp_again == 2:
    from process_pfs_only import post_proc_pfs_only 
    resp_again = repeat_loop_proc()

    if resp_again == 2:
    
        resTypes = ['Full HD',
                    'HD+',
                    'HD',
                    'qHD',
                    'nHD',
                    '960H',
                    'HVGA',
                    'VGA', 
                    'SVGA',
                    'DVGA',
                    'QVGA',
                    'QQVGA',
                    'HQVGA'
                   ]

        res_dims = ['(1920x1080)',
                    '(1600x900)',
                    '(1280x720)',
                    '(960x540)',
                    '(640x360)',
                    '(960x480)',
                    '(480x320)',
                    '(640x480)',
                    '(800x600)',
                    '(960x640)',
                    '(320x240)',
                    '(160x120)',
                    '(240x160)'
                  ]   

        res_fullTypes = []

        for ind_r, r in enumerate(resTypes):
            r += ' ' + res_dims[ind_r]
            res_fullTypes.append(r)


        basler_check = False

        basler_check, model_name = confirm_basler() 

        if basler_check == True:
            
            from pfs_input_acq_step_only import read_pfs_file
            
            repeat = True  
            
            while True:
                control_inf = exp_control()
                
                if control_inf[0] == True:
                    
                    numberTests = control_inf[1]
                    gap_bet_tests = control_inf[2]
                    
                    break
                else:
                    continue 
            
            print("Number of tests: " + str(numberTests))
                
            if numberTests > 1: 
                print("Going to perform for " + str(numberTests) + " tests with a " + str(gap_bet_tests) + " minutes gap")
            else:
                print("Going to perform just 1 test")
            
            print("numTests init: " + str(numberTests))
            
            enab = optional_prop()
             
            enab_val_pfs = enab[0]
            enab_timestamps = enab[1]
            
            gap_bet_tests*=60 
            
            read_pfs_file(gap_bet_tests, numberTests, repeat, startTime, enab_val_pfs, enab_timestamps)