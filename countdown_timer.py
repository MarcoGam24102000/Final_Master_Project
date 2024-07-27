# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:47:54 2022

@author: Rui Pinto
"""

import time
import PySimpleGUI as sg

def countdown_timer_display(t):
    
    t *= 60
    initial_t = t
    
    t1 = time.time()
    
    name_window = 'Time left'

    layout = [[sg.Text('', key='-TIME_LEFT-')],
              [sg.Button('Exit')]]

    window = sg.Window(name_window, layout, finalize=True)

    while t >= 0 and time.time()-t1 <= initial_t:
        
        event, values = window.read(timeout=1000)
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        window['-TIME_LEFT-'].update(f'Time left: {int(t)} seconds')
        t -= 1

    window.close()
    return True
  

 