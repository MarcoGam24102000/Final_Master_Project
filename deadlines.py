# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 22:25:29 2022

@author: marco
"""


def time_configs():
    
    import PySimpleGUI as sg
    
    default_sec_min = 60
    default_min_hour = 60
    default_hour_day = 24
    default_days_week = 7
    default_days_month = 30
    default_days_year = 365 
    
    change_det = False 
    counts = []
    
    changes_timing = {}
    
    layout = [        
        [sg.Text('Nº de segundos por minuto: '), sg.Input(default_text= str(default_sec_min), size=(19, 1), key="SEC_MIN")], 
        [sg.Text('Nº de minutos por hora: '), sg.Input(default_text= str(default_min_hour), size=(19, 1), key="MIN_HOUR")],
        [sg.Text('Nº de horas por dia: '), sg.Input(default_text= str(default_hour_day), size=(19, 1), key="HOUR_DAY")],
        [sg.Text('Nº de dias por semana: '), sg.Input(default_text= str(default_days_week), size=(19, 1), key="DAY_WEEK")],
        [sg.Text('Nº de dias por mês: '), sg.Input(default_text= str(default_days_month), size=(19, 1), key="DAY_MONTH")],
        [sg.Text('Nº de dias por ano: '), sg.Input(default_text= str(default_days_year), size=(19, 1), key="DAY_YEAR")],
        [sg.Button('Back'), sg.Button('Save')]        
    ]
    
    
    window = sg.Window("Time configurations (for deadline considerations only)", layout) 

    
    while True:
       event, values = window.read()
       if event == "Exit" or event == sg.WIN_CLOSED: 
           break
       if event == "Back":
           break
       if event == "Save":
           
           changes_timing['NOT_NULL'] = 1
           
           if int(values['SEC_MIN']) != default_sec_min:
               change_det = True
               counts.append(0)
               default_sec_min = int(values['SEC_MIN'])
           if int(values['MIN_HOUR']) != default_min_hour:   
               change_det = True
               counts.append(1)
               default_min_hour = int(values['MIN_HOUR'])
           if int(values['HOUR_DAY']) != default_hour_day:  
               change_det = True
               counts.append(2)
               default_hour_day = int(values['HOUR_DAY'])
           if int(values['DAY_WEEK']) != default_days_week: 
               change_det = True
               counts.append(3)
               default_days_week = int(values['DAY_WEEK'])
           if int(values['DAY_MONTH']) != default_days_month:  
               change_det = True
               counts.append(4)
               default_days_month = int(values['DAY_MONTH']) 
           if int(values['DAY_YEAR']) != default_days_year: 
               change_det = True 
               counts.append(5)
               default_days_year = int(values['DAY_YEAR'])
               
  #         break
       
        
       window.close()
       
       if change_det == True:
           
           changes_timing = {}
           
           for count_timing_param in counts:
               if count_timing_param == 0:
                   changes_timing['SEC_MIN'] = default_sec_min
               elif count_timing_param == 1:
                   changes_timing['MIN_HOUR'] = default_min_hour
               elif count_timing_param == 2:
                   changes_timing['HOUR_DAY'] = default_hour_day
               elif count_timing_param == 3:
                   changes_timing['DAY_WEEK'] = default_days_week
               elif count_timing_param == 4:
                   changes_timing['DAY_MONTH'] = default_days_month
               elif count_timing_param == 5:
                   changes_timing['DAY_YEAR'] = default_days_year 
       
       else:
            print("Nothing changes, related to timing configurations related to deadline considerations")
            
       window.close()           
       
       return changes_timing 
    

def deadline_pfs_file_gui(time_guidelines):

    default_sec_min = time_guidelines[0]
    default_min_hour = time_guidelines[1]
    default_hour_day = time_guidelines[2]
    default_days_week = time_guidelines[3]
    default_days_month = time_guidelines[4] 
    default_days_year = time_guidelines[5]
    
    import PySimpleGUI as sg
    
    period_exp_pfs_file_seconds = 0
    
    items_standard_timeperiods = ['1 hour', '1 day', '1 week', '1 month', '2 months', '3 months', '4 months', '5 months', '6 months', 
                                  '7 months', '8 months', '9 months', '10 months', '11 months', '1 year', '2 years', '3 years', '4 years', '5 years']
    
    layout = [
        
        [sg.Text('PFS file going to be expired in less than : ')], 
        [sg.InputCombo(items_standard_timeperiods, size=(19, 1), key="PERIOD_EXPIR_PFS_FILE", change_submits=False)],
        [sg.Button('Back'), sg.Button('Next')]
        
    ]  
 
     
    window = sg.Window("Deadline for '.pfs' file", layout) 

    
    while True:
       event, values = window.read()
       if event == "Exit" or event == sg.WIN_CLOSED:
           break
       if event == "Back":
           break
       if event == "Next":
           period_expir_pfs = values['PERIOD_EXPIR_PFS_FILE']
           
           if 'hour' in period_expir_pfs or 'hours' in period_expir_pfs:
               period = period_expir_pfs.split(' ')
               hours = int(period[0])
               period_exp_pfs_file_seconds = hours*(default_sec_min*default_min_hour)
           elif 'day' in period_expir_pfs or 'days' in period_expir_pfs:
               period = period_expir_pfs.split(' ')
               days = int(period[0])
               period_exp_pfs_file_seconds = days*(default_hour_day*(default_sec_min*default_min_hour))
           elif 'week' in period_expir_pfs or 'weeks' in period_expir_pfs:
               period = period_expir_pfs.split(' ')
               weeks = int(period[0])
               period_exp_pfs_file_seconds = weeks*(default_days_week*default_hour_day*(default_sec_min*default_min_hour))
           elif 'month' in period_expir_pfs or 'months' in period_expir_pfs:
               period = period_expir_pfs.split(' ')
               months = int(period[0])
               period_exp_pfs_file_seconds = months*(default_days_month*default_hour_day*(default_sec_min*default_min_hour))       ## 30 days per months on average
           elif 'year' in period_expir_pfs or 'years' in period_expir_pfs:
               period = period_expir_pfs.split(' ')
               years = int(period[0])
               period_exp_pfs_file_seconds = years*(default_days_year*default_hour_day*(default_sec_min*default_min_hour))       ## 365 days by default   
               
               ## gui for time specifications (hours per day, days per months, days per year, ...)
               
           break
    
    
    window.close()     
    
    return period_exp_pfs_file_seconds

    
## max_time_for_pfs_file_expir = deadline_pfs_file_gui()
## print("PFS file is going to be expired, at maximum, in " + str(max_time_for_pfs_file_expir) + " seconds") 

## changes_timing = time_configs()


    
