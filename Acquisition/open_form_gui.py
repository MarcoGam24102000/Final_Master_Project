# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 00:24:56 2022

@author: marco
"""

def exp_control():

    import PySimpleGUI as sg
    import time
    
    numberMaxTests = 20    
    
    save_cmd = False
    
    valuesNumberTests = []
    for i in range(0,numberMaxTests + 1):
        valuesNumberTests.append(str(i))
        
    gapBetTests = []
    timeBase = 5    ## minutes
    
    for i in range(1,13):
        gapBetTests.append(str(timeBase*i) + " minutes")         
    
    for i in range(1,13):
        gapBetTests.append(str(timeBase*i) + " seconds")         
     
    
    col = sg.Column([     
        [sg.Frame(layout=[[sg.Text('Number of tests:')],              ## Add a '+' to increase the number of tests until a maximum of 20, or a '-' to decrease the number of tests
                          [sg.Combo(valuesNumberTests, default_value = '1', key = "NumberTests")],
                          [sg.Text('Gap between tests:', key = 'GapText', size=(20, 2), visible=False)],
                          [sg.Combo(gapBetTests, default_value = '30 minutes', size=(20, 2), key = "GapBetTests", visible=False)]
           ##               [sg.Text('minutes', size=(20, 2), key = 'MinutesText', visible=False)]                                        
                          ], title='Setup form:')]    
    ])
    
    
    layout = [ [col],     
            # Actions Frame 
            [sg.Frame(layout=[[sg.Button('Save'), sg.Button('Next')                 
                ]], title='Actions:')]]        ## , sg.Button('Clear'), sg.Button('Delete'),      
    window = sg.Window('GUI', layout, disable_close=True, resizable = True, finalize = True, margins=(0,0))  
    
    
    numberTests = 0
    gap_bet_tests = 0
    
    control_setup_exp = (0,0)
    
    nextOpt = False
    
 ##   res = 0    
    
    while True:           
            event, values = window.read()
            print(event, values) 
            
            if event == "Exit" or event == sg.WIN_CLOSED:                
                save_cmd = False               
                break
            
            if event == "Next":
                numberTests = int(values['NumberTests'])
                
                if numberTests == 1:
                    gap_bet_tests = 0
                    
                else:       
                    
                    window['GapText'].update(visible=True)
                    window['GapBetTests'].update(visible=True)                   
             #       window['MinutesText'].update(visible=True)
                    nextOpt = True 
                    
             
            if event == 'Save':
                print("OK")
                
                if nextOpt == False:
                
                    numberTests = int(values['NumberTests'])                   
                    
                
                    if numberTests == 1:
                        gap_bet_tests = 0
                
                if nextOpt == True: 
                    
                    # windowx = sg.Window('GUI', layout, disable_close=True, resizable = True, finalize = True, margins=(0,0))  
                    
                    # while True:           
                    #         event, values = windowx.read()
                    #         print(event, values) 
                    
                            gap_bet_tests = 0
                            gap_bet_tests_str = values['GapBetTests']
                            
                            if "seconds" in gap_bet_tests_str:
                                gap_parts = gap_bet_tests_str.split(' ')
                                for p in gap_parts:
                                    if p.isdigit():
                                        gap_bet_tests = round(int(p)/60,5)
                                        break
                            elif "minutes" in gap_bet_tests_str:                                
                                gap_parts = gap_bet_tests_str.split(' ')
                                for p in gap_parts:
                                    if p.isdigit():
                                        gap_bet_tests = int(p) 
                                
                    #        windowx.close()
 
                window.close()              
    

    print("Amount of acquisition time: " + str(numberTests * gap_bet_tests - gap_bet_tests))
                         
    save_cmd = True
    
    control_setup_exp = (save_cmd, numberTests, gap_bet_tests)      
     
    return control_setup_exp           
                
   
            
            
        
        
    


