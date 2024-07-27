# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 09:35:46 2022

@author: marco
"""

# str_file = "out20_12_1parameters__201222_093058__full"

# folder_name = "out20_12_1"
# inter_str_par = "parameters_"
# full_str = "_full"

# str_file_simple = "out19_12_1parameters__191222_093135_"
# folder_name_simple = "out19_12_1"
# inter_str_par_simple = "parameters_"


def dateAndTime_from_pfs_file_full(str_file, folder_name, inter_str_par, full_str):

    str_sec = str_file.split(folder_name)
    rem_str_sec = str_sec[1]
    
    str_third = rem_str_sec.split(inter_str_par)
    rem_str_third = str_third[1] 
    
    str_four = rem_str_third.split(full_str)
    rem_str_four = str_four[0]
    
    dateAndTime = rem_str_four.split('_') 
    
    date_str = dateAndTime[1]
    time_str = dateAndTime[2]
    
    couple_info_dateTime = (date_str, time_str)
    
    return couple_info_dateTime


def dateAndTime_from_pfs_file(str_file, folder_name, inter_str_par):
    
    
    resp_gen_basler_software = input("The provided .pfs file was generated using this software ? ")
    
    if ( 's' in resp_gen_basler_software) or ('S' in resp_gen_basler_software) or ('y' in resp_gen_basler_software) or ('Y' in resp_gen_basler_software):
    
        print(str_file)
        print(folder_name)
        print(inter_str_par)
    
        str_sec = str_file.split(folder_name)
        rem_str_sec = str_sec[1]   
        
        str_third = rem_str_sec.split(inter_str_par)
        rem_str_third = str_third[1]    
        
        dateAndTime = rem_str_third.split('_') 
        
        date_str = dateAndTime[1]
        time_str = dateAndTime[2]
    
    else:
        from datetime import datetime
        
        date_apart = str(datetime.now()).split(' ')
        
        y = date_apart[1]
         
        y_s = y.split(':')
        
        time_str = y_s[0] + y_s[1] + y_s[2]
        
        x = date_apart[0]
        
  ##    x = datetime.date.today()
        x_s = str(x).split('-')        
        date_str = x_s[2] + x_s[1] + x_s[0].split('20')[1]
    
    couple_info_dateTime = (date_str, time_str) 
    
    return couple_info_dateTime 

# couple_info_dateTime = dateAndTime_from_pfs_file(str_file_simple, folder_name_simple, inter_str_par_simple)
 
# couple_info_dateTime_full = dateAndTime_from_pfs_file_full(str_file, folder_name, inter_str_par, full_str)

