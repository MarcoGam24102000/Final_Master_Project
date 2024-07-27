# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 12:05:45 2023

@author: Rui Pinto
"""


def com_tests_panel(number_tests):
    import PySimpleGUI as sg
    
    tests_list = []
    
    
    for x in range(0, number_tests):
        tests_list.append(x+1)
    
    layout= [        
        [sg.Text("Initial test number"), sg.InputCombo(tests_list, size = (19, 1), key="FIRST_TEST", change_submits=False)],
        [sg.Text("Final test number"), sg.InputCombo(tests_list, size = (19, 1), key="SEC_TEST", change_submits=False)],
        [sg.Button("Exit"), sg.Button("Next")]
    ]
    
    
    window = sg.Window("Inter-tests comparison processing", layout)
    
    again = True
    
    pair_tests = (0,0)
    
    while again:
        event, values = window.read()
        
        if event == sg.WIN_CLOSED or event == "Exit":
            break
        
        if event == "Next":
            if int(values['FIRST_TEST']) != int(values['SEC_TEST']):
                again = False
                pair_tests = (int(values['FIRST_TEST']), int(values['SEC_TEST']))
                break
            else:
                print("First test number should be different from the second one")
    
    window.close()    
    
    return pair_tests 

pair_tests = com_tests_panel(5)