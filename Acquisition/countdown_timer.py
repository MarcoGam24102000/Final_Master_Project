# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:47:54 2022

@author: Rui Pinto
"""

def countdown_timer_display(t):
    import time
    import PySimpleGUI as sg
    
    t *= 60
    
    print("Time to wait: " + str(t))
  #  import sys
  #  sys.exit()
   
    # define the countdown func.
    def countdown(t):
        
        readyForNext = False 
         
        while t >= 0:      
                
             if True:
                print("Counting down ...")
                
                time_left = sg.Text(str(t), size=(19, 1), enable_events=True, key="TIME_LEFT")
                   
                col = sg.Column([     
                       [sg.Frame(layout=[[time_left]], title='Time left:' )]])
                   
                next_flag = False
                    
                layout = [ [col] ]   
                
                window = sg.Window('Time left', layout, disable_close=True, finalize = True)  
                
                button,values = window.read(timeout=1000)              
                
                print(t)
                
                t -= 1         
              
                window.close()
                
                if t == -1:
                    
                    readyForNext= True
                    break
                     
        return readyForNext           

    next_flag = countdown(int(t))
    
    return next_flag 
  

 